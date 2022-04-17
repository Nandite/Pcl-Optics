// Copyright (c) 2020 Papa Libasse Sow.
// https://github.com/Nandite/Pcl-Optics
// Distributed under the MIT Software License (X11 license).
//
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>
#include <random>
#include "Optics.hpp"

template <typename PointT>
bool planeExtractor(const typename pcl::PointCloud<PointT>::Ptr& source, const pcl::PointIndices::Ptr& indices,
                    pcl::PointIndices::Ptr& inliers, pcl::PointIndices::Ptr& outliers,
                    const unsigned int maxSegment = 2, const double threshold = 0.02f,
                    const unsigned int maxIteration = 100) {
  if (source->empty()) {
    return false;
  }
  typename pcl::PointCloud<PointT>::Ptr bufferCloud(new pcl::PointCloud<PointT>);
  typename pcl::PointCloud<PointT>::Ptr extractedFeaturesCloud(new pcl::PointCloud<PointT>);
  pcl::PointIndices::Ptr extractedIndices(new pcl::PointIndices());
  pcl::ModelCoefficients::Ptr stubCoefficients(new pcl::ModelCoefficients());
  pcl::SACSegmentation<PointT> sacSegmentation{};
  sacSegmentation.setOptimizeCoefficients(true);
  sacSegmentation.setModelType(pcl::SACMODEL_PLANE);
  sacSegmentation.setMethodType(pcl::SAC_RANSAC);
  sacSegmentation.setMaxIterations(maxIteration);
  sacSegmentation.setDistanceThreshold(threshold);
  sacSegmentation.setInputCloud(source);
  outliers->indices.insert(std::begin(outliers->indices), std::begin(indices->indices), std::end(indices->indices));
  // For each iteration of the loop we have R(n) = Io - sum(E(i))[i in the range 0..n]
  // R(i) : the remaining point after the segmentation at the stage i
  // Io the initial number of points to segment
  // E(n) the extracted points after each segmentation loop
  // We can hence write that
  // R(n) = R(n-1) - E(n) -> the remaining point at the stage n is equals to the
  // remaining points at the previous stage (n-1) minus the extracted point at the
  // stage (n).
  // We can therefore deduce that I(n) = R(n-1) -> the input points at the stage (n)
  // for segmentation is simply that remaining points at the stage (n-1). this relation
  // can also be writen I(n+1) = R(n) which is clearer for this loop application.
  for (auto segmentCount{0u}; segmentCount < maxSegment; ++segmentCount) {
    // Computation of E(n), extracted point at this loop stage
    sacSegmentation.setIndices(outliers);
    sacSegmentation.segment(*extractedIndices, *stubCoefficients);
    if (extractedIndices->indices.empty()) {
      break;
    }

    // Accumulation stage : sum(E(i))[i in the range 0..n]
    inliers->indices.insert(std::end(inliers->indices), std::begin(extractedIndices->indices),
                            std::end(extractedIndices->indices));
    std::sort(std::begin(inliers->indices), std::end(inliers->indices));

    outliers->indices.clear();
    std::set_difference(std::begin(indices->indices), std::end(indices->indices), std::begin(inliers->indices),
                        std::end(inliers->indices),
                        std::back_inserter(outliers->indices)  // I(n+1) = R(n)
    );
  }

  return true;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage : " << argv[0] << " <filename>" << std::endl;
    return EXIT_FAILURE;
  }

  std::mt19937 t(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<unsigned int> rgb(1, 255);

  pcl::PCDReader reader{};
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>),
      downSampledCloud(new pcl::PointCloud<pcl::PointXYZ>), sceneCloud(new pcl::PointCloud<pcl::PointXYZ>);
  reader.read(argv[1], *cloud);
  std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl;

  pcl::VoxelGrid<pcl::PointXYZ> vg{};
  vg.setInputCloud(cloud);
  vg.setLeafSize(0.01f, 0.01f, 0.01f);
  vg.filter(*downSampledCloud);
  std::cout << "PointCloud after filtering has: " << downSampledCloud->points.size() << " data points." << std::endl;

  const auto nbDownSampledPoints{downSampledCloud->size()};
  pcl::PointIndicesPtr downSampledPointsIndices{new pcl::PointIndices};
  {
    std::vector<int> indices(nbDownSampledPoints);
    std::iota(std::begin(indices), std::end(indices), 0);
    downSampledPointsIndices->indices.swap(indices);
  }
  pcl::PointIndicesPtr extractionInliers{new pcl::PointIndices};
  pcl::PointIndicesPtr extractionOutliers{new pcl::PointIndices};
  if (!planeExtractor<pcl::PointXYZ>(downSampledCloud, downSampledPointsIndices, extractionInliers,
                                     extractionOutliers)) {
    std::clog << "Failed to extract the planes representing the scene." << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<pcl::PointIndicesPtr> clusters{};
  {
    pcl::IndicesPtr clusterIndices{std::make_shared<std::vector<int>>(extractionOutliers->indices)};
    const auto start{std::chrono::steady_clock::now()};
    if (!Optics::optics<pcl::PointXYZ>(downSampledCloud, clusterIndices, 10, 0.05, clusters)) {
      std::clog << "Failed to apply the optics to the input cloud" << std::endl;
      return EXIT_FAILURE;
    }
    const auto end{std::chrono::steady_clock::now()};
    const auto duration{std::chrono::duration_cast<std::chrono::milliseconds>(end - start)};
    std::cout << "Optics clustering of [" << clusterIndices->size() << "] points took [" << duration.count() << "] ms."
              << std::endl;
  }
  int v0{0}, v1{1}, v2{2};
  pcl::visualization::PCLVisualizer viewer("Visualizer");
  viewer.setSize(1280, 1024);
  viewer.setShowFPS(true);

  viewer.createViewPort(0.0, 0.0, 0.60, 0.5, v0);
  viewer.setBackgroundColor(0.3, 0.3, 0.3, v0);
  viewer.createViewPort(0.0, 0.5, 0.60, 1.0, v1);
  viewer.setBackgroundColor(0.0, 0.0, 0.0, v1);
  viewer.createViewPort(0.60, 0.0, 1.0, 1.0, v2);
  viewer.setBackgroundColor(0.0, 0.0, 0.0, v2);

  viewer.addText("Input down-sampled cloud", 0, 0, "Input cloud Id", v0);
  viewer.addText("Clusters", 0, 0, "Clusters Id", v1);
  viewer.addText("Indices", 0, 0, "Indices Id", v2);
  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(downSampledCloud, rgb(t), rgb(t), rgb(t));
    viewer.addPointCloud(downSampledCloud, color, "downSampledCloud", v0);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "downSampledCloud", v0);
  }

  {
    pcl::ExtractIndices<pcl::PointXYZ> extract{};
    extract.setInputCloud(downSampledCloud);
    extract.setIndices(extractionOutliers);
    extract.setNegative(false);
    extract.filter(*sceneCloud);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(sceneCloud, rgb(t), rgb(t), rgb(t));
    viewer.addPointCloud(sceneCloud, color, "sceneCloud", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sceneCloud", v2);
  }

  auto id{0u};
  for (const auto& cluster : clusters) {
    if (cluster->indices.size() < 10) continue;

    std::cout << "Cluster " << id << " size is : " << cluster->indices.size() << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& index : cluster->indices) {
      clusterCloud->push_back((*downSampledCloud)[index]);
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(clusterCloud, rgb(t), rgb(t), rgb(t));
    std::string strid{"cloud_cluster_" + std::to_string(id++)};
    viewer.addPointCloud(clusterCloud, color, strid, v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, strid, v1);
  }

  while (!viewer.wasStopped()) {
    viewer.spinOnce();
  }

  return EXIT_SUCCESS;
}
