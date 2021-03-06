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

int main(int argc, char* argv[]) {

    if(argc < 2)
    {
        std::cerr << "Usage : " << argv[0] << " <filename>" << std::endl;
        return EXIT_FAILURE;
    }

  std::mt19937 t(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<unsigned int> rgb(1, 255);

  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>),
      bufferCloud(new pcl::PointCloud<pcl::PointXYZ>);
  reader.read(argv[1], *cloud);
  std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl;

  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud(cloud);
  vg.setLeafSize(0.01f, 0.01f, 0.01f);
  vg.filter(*filteredCloud);
  std::cout << "PointCloud after filtering has: " << filteredCloud->points.size() << " data points." << std::endl;

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.02);

  int nr_points = (int)filteredCloud->points.size();
  while (filteredCloud->points.size() > 0.3 * nr_points) {
    seg.setInputCloud(filteredCloud);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.empty()) {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(filteredCloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points."
              << std::endl;
    extract.setNegative(true);
    extract.filter(*bufferCloud);
    *filteredCloud = *bufferCloud;
  }

  std::vector<pcl::PointIndicesPtr> clusters;
  Optics::optics<pcl::PointXYZ>(filteredCloud, 10, 0.05, clusters);

  pcl::visualization::PCLVisualizer viewer("Visualizer");
  viewer.setSize(1280, 1024);
  viewer.setShowFPS(true);

  unsigned int id = 0;
  for (const auto& c : clusters) {
    if (c->indices.size() < 10) continue;

    std::cout << "Cluster " << id << " size is : " << c->indices.size() << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& index : c->indices) {
      clusterCloud->push_back((*filteredCloud)[index]);
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(clusterCloud, rgb(t), rgb(t), rgb(t));
    std::string strid = "cloud_cluster_" + std::to_string(id++);
    viewer.addPointCloud(clusterCloud, color, strid);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, strid);
  }

  while (!viewer.wasStopped()) {
    viewer.spinOnce();
  }

  return EXIT_SUCCESS;
}
