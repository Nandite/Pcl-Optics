// Copyright (c) 2020 Papa Libasse Sow.
// https://github.com/Nandite/Pcl-Optics
// Distributed under the MIT Software License (X11 license).
//
// Copyright (c) 2016 Ingo Proff.
// https://github.com/CrikeeIP/OPTICS-Clustering
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

#ifndef OPTICS_HPP
#define OPTICS_HPP

#include <pcl/common/geometry.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>

namespace Optics {

/**
 * @brief Handler structure storing a point of space DS to cluster and its associated reachability distance.
 */
struct ReachabilityDistance {
  /**
   * @param pointIndex index of the point within the cloud to cluster.
   * @param reachabilityDistance Reachability distance of the point.
   */
  ReachabilityDistance(const int pointIndex, const double reachabilityDistance)
      : pointIndex(pointIndex), reachabilityDistance(reachabilityDistance) {}

  int pointIndex;
  double reachabilityDistance;
};

/**
 * Less than operator to compare two ReachabilityDistance instances.
 * @param lhs left-hand side instance to compare with.
 * @param rhs right-hand side instance to compare with.
 * @return:
 * - If the reachability distance of both instances are equals:
 *      - True if the index of lhs is strictly less than the index of rhs.
 *      - False otherwise.
 * - Otherwise, True if the reachability distance of lhs is strictly less than the
 *   reachability distance of rhs, else False.
 */
inline bool operator<(const ReachabilityDistance& lhs, const ReachabilityDistance& rhs) {
  return (lhs.reachabilityDistance <= rhs.reachabilityDistance && lhs.reachabilityDistance >= rhs.reachabilityDistance)
             ? (lhs.pointIndex < rhs.pointIndex)
             : (lhs.reachabilityDistance < rhs.reachabilityDistance);
}

/**
 * Equality operator for comparing two ReachabilityDistance instances.
 * @param lhs left-hand side instance to compare with.
 * @param rhs right-hand side instance to compare with.
 * @return:
 * - True if the reachability distance fields are equals and the index fields are also equals
 * - False otherwise.
 */
inline bool operator==(const ReachabilityDistance& lhs, const ReachabilityDistance& rhs) {
  return (lhs.reachabilityDistance <= rhs.reachabilityDistance &&
          lhs.reachabilityDistance >= rhs.reachabilityDistance) &&
         (lhs.pointIndex == rhs.pointIndex);
}

namespace internals {

/**
 * @brief Compare two pointers by de-referencing them and actually compare the
 * values they point to.
 * @tparam T Type of the pointed to compare.
 * @param l left-hand side
 * @param r right-hand side
 * @return True if de-referenced left is strictly less than de-referenced right,
 * False otherwise.
 */
template <class T>
bool dereferenceLess(T const* l, T const* r) {
  return *l < *r;
}

/**
 * @brief Verifies that a vector does not contain duplicate values. The vector
 * to check remain unchanged. This function perform the check with the worst case
 * complexity of O(n*log(n))
 * @tparam T Type held by the vector.
 * @param x Vector to check for uniqueness of values.
 * @return True if every value in the vector are different, False otherwise.
 */
template <class T>
bool isUnique(std::vector<T> const& x) {
  std::vector<T const*> pointers{};
  pointers.reserve(x.size());
  for (auto i{0u}; i < x.size(); ++i) {
    pointers.push_back(&x[i]);
  }
  std::sort(std::begin(pointers), std::end(pointers),
            std::ptr_fun(&dereferenceLess<T>));  // O(N log N)
  return adjacent_find(std::begin(pointers), std::end(pointers), std::not2(std::ptr_fun(&dereferenceLess<T>))) ==
         std::end(pointers);
}

/**
 * @brief Structure permitting to compare two values given a custom transformation.
 * @tparam F The type of the transformation.
 */
template <typename F>
struct isLessByStruct {
  /**
   * @brief Ctor.
   * @param f The transformation to apply during the comparison.
   */
  explicit isLessByStruct(F f) : transformation(f){};

  /**
   * @brief Compare two values given a transformation f.
   * @tparam T The type of the value to compare.
   * @param x left-hand side value
   * @param y right-hand side value
   * @return True if f(x) < f(y), False otherwise.
   */
  template <typename T>
  bool operator()(const T& x, const T& y) {
    return transformation(x) < transformation(y);
  }

 private:
  F transformation;
};

/**
 * @brief Find the nth element of a sequence given a comparator.
 * @tparam Compare The type of the comparator to use.
 * @tparam Container The type of the container to use.
 * @tparam T The type of the value held by the container.
 * @param comp The comparator to use.
 * @param n The nth largest element to find.
 * @param xs The container to search the element for.
 * @return The nth largest element of the sequence.
 */
template <typename Compare, typename Container, typename T = typename Container::value_type>
T nthElementBy(Compare comp, const std::size_t n, const Container& xs) {
  auto result{xs};
  auto nth{std::begin(result)};
  std::advance(nth, n);
  std::nth_element(std::begin(result), nth, std::end(result), comp);
  return *nth;
}

/**
 * @brief Find the nth largest element of a sequence given a transformation.
 * @tparam F The type of transformation to use to find the element.
 * @tparam Container The type of the container to use.
 * @tparam T The type of the value held by the container.
 * @param f Instance of the transformation to use.
 * @param n The nth largest element to find.
 * @param xs The container to search the element for.
 * @return The nth largest element of the sequence.
 */
template <typename F, typename Container, typename T = typename Container::value_type>
T nthLargestElementOn(F f, const std::size_t n, const Container& xs) {
  return nthElementBy(isLessByStruct<F>(f), n, xs);
}

/**
 * @brief Find two points min and max such as all the point from the given input cloud reside within the
 * bounding box going through both points.
 * @param points The cloud to find a bounding box for.
 * @return The min and max point of the box.
 */
template <typename PointT>
std::pair<PointT, PointT> boundingBox(const typename pcl::PointCloud<PointT>::Ptr& points) {
  PointT min{(*points)[0]};
  PointT max{(*points)[1]};

  for (const auto& p : *points) {
    min.x = p.x < min.x ? p.x : min.x;
    max.x = p.x > max.x ? p.x : max.x;

    min.y = p.y < min.y ? p.y : min.y;
    max.y = p.y > max.y ? p.y : max.y;

    min.z = p.z < min.z ? p.z : min.z;
    max.z = p.z > max.z ? p.z : max.z;
  }

  return {{min}, {max}};
}

/**
 * @brief Compute the core distance of a point from the input cloud. The core distance is the minimum radius distance
 * needed to classify a point as a core point.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param point The point to compute the core distance for.
 * @param points The inputs cloud to cluster.
 * @param neighborIndices Indices of the neighbors of the point to compute the core distance for.
 * @param minPts Minimal number of point needed to qualify a point as a core point.
 * @param coreDistance The core distance computed for the point.
 * @return True if the core distance of the point is defined, false otherwise.
 */
template <typename PointT>
bool computeCoreDistance(const PointT& point, const typename pcl::PointCloud<PointT>::Ptr& points,
                         const std::vector<int>& neighborIndices, std::size_t minPts, double& coreDistance) {
  if (neighborIndices.size() < minPts) {
    return false;
  }

  auto coreElementIndex{nthLargestElementOn(
      [&points, &point](int idx) -> double { return pcl::geometry::squaredDistance(point, (*points)[idx]); },
      minPts - 1, neighborIndices)};
  coreDistance = pcl::geometry::distance(point, (*points)[coreElementIndex]);
  return true;
}

/**
 * @brief Expand a cluster by computing the reachability of non-processed points located into the epsilon neighborhood
 * of a core point.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param point The core point to expand the cluster around.
 * @param points The inputs cloud to cluster.
 * @param neighborIndices indices of the core point neighbors.
 * @param coreDistance Core distance of the point to expand the cluster around.
 * @param states Sequence containing the processing state of the points.
 * @param reachabilities Sequence containing the reachabilities of the input cloud points.
 * @param seeds Sequence into which new points found through the expansion are inserted for future processing.
 */
template <typename PointT, typename BoolSequence, typename ReachabilitySequence>
void expandCluster(const PointT& point, const typename pcl::PointCloud<PointT>::Ptr& points,
                   const std::vector<int>& neighborIndices, const double coreDistance, const BoolSequence& states,
                   ReachabilitySequence& reachabilities, std::set<Optics::ReachabilityDistance>& seeds) {
  for (const auto& neighbor : neighborIndices) {
    const auto hasBeenProcessed{states.at(neighbor)};
    if (hasBeenProcessed) {
      continue;
    }
    const auto newReachabilityDistance{
        std::max(coreDistance, double(pcl::geometry::distance(point, (*points)[neighbor])))};
    if (reachabilities[neighbor] < 0.0) {
      reachabilities[neighbor] = newReachabilityDistance;
      seeds.insert({neighbor, newReachabilityDistance});
    } else if (newReachabilityDistance < reachabilities[neighbor]) {
      // erase from seeds
      seeds.erase({neighbor, reachabilities[neighbor]});
      // update reachability
      reachabilities[neighbor] = newReachabilityDistance;
      // re-insert seed with new reachability
      seeds.insert({neighbor, newReachabilityDistance});
    }
  }
}
}  // namespace internals

/**
 * @brief Perform an estimation of an epsilon to be used as a parameter to the Optics algorithm for a given input
 * cloud of point. Large values of epsilon can result in longer execution time (O(n^2) in worst case) as each
 * neighborhood query could return the entire database. Low value of epsilon could on the other side causes
 * the reachability and core distance to be undefined if the clusters are not sufficiently dense. This method is
 * a heuristic which attempt to find sufficiently large value of epsilon. It does so by make the assumption that
 * the points are randomly distributed within the space DS and uses the K-Nearest Neighbors (KNN) distance to determine
 * an epsilon which is the radius of the 3-d dimensional hypersphere S in the space DS such as all the k points
 * (k = minPts) lies within S.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param points The cloud to compute an epsilon estimation for.
 * @param minPts Minimal number of point per cluster.
 * @return An estimation for the epsilon parameter or zero if the cloud size is less than or equal to 1.
 */
template <typename PointT>
double epsilonEstimation(const typename pcl::PointCloud<PointT>::Ptr& points, const std::size_t minPts) {
  const auto size{points->size()};

  if (size <= 1) {
    return 0;
  }

  const auto dimension{3};
  const auto space{Optics::internals::boundingBox<PointT>(points)};
  const auto spaceVolume{std::abs(double(space.second.x - space.first.x)) *
                         std::abs(double(space.second.y - space.first.y)) *
                         std::abs(double(space.second.z - space.first.z))};

  const auto spacePerMinPtsPoints{(spaceVolume / static_cast<double>(size)) * static_cast<double>(minPts)};
  const auto nDimUnitBallVol{std::sqrt(std::pow(M_PI, dimension)) / std::tgamma(dimension / 2.0 + 1.0)};
  return std::pow(spacePerMinPtsPoints / nDimUnitBallVol, 1.0 / dimension);
}

/**
 * @brief Perform an estimation of an epsilon to be used as a parameter to the Optics algorithm for a given input
 * 2 dimensions cloud of point. Large values of epsilon can result in longer execution time (O(n^2) in worst case)
 * as each neighborhood query could return the entire database. Low value of epsilon could on the other side causes
 * the reachability and core distance to be undefined if the clusters are not sufficiently dense. This method is
 * a heuristic which attempt to find sufficiently large value of epsilon. It does so by make the assumption that
 * the points are randomly distributed within the space DS and uses the K-Nearest Neighbors (KNN) distance to determine
 * an epsilon which is the radius of the 2-d dimensional hypersphere S in the space DS such as all the k points
 * (k = minPts) lies within S.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param points The cloud to compute an epsilon estimation for.
 * @param minPts Minimal number of point per cluster.
 * @return An estimation for the epsilon parameter or zero if the cloud size is less than or equal to 1.
 */
template <typename PointT>
double epsilon2dEstimation(const typename pcl::PointCloud<PointT>::Ptr& points, const std::size_t minPts) {
  const auto size{points->size()};

  if (size <= 1) {
    return 0;
  }

  const auto dimension{2};
  const auto space{Optics::internals::boundingBox<PointT>(points)};
  const auto spaceVolume{std::abs(double(space.second.x - space.first.x)) *
                         std::abs(double(space.second.y - space.first.y))};

  const auto spacePerMinPtsPoints{(spaceVolume / static_cast<double>(size)) * static_cast<double>(minPts)};
  const auto nDimUnitBallVol{std::sqrt(std::pow(M_PI, dimension)) / std::tgamma(dimension / 2.0 + 1.0)};
  return std::pow(spacePerMinPtsPoints / nDimUnitBallVol, 1.0 / dimension);
}

/**
 * @brief Given an ordered sequence of reachability distances for a given input cloud, this method outputs a set
 * of indices sequence, each one containing the indices of all point belonging to a cluster.
 * @param reachabilityDistances Reachability distances computed for a given cloud of point.
 * @param reachabilityThreshold Maximal reachability distance allowed for a point p to belong to a cluster being
 * formed. If a point with a reachability distance higher than the threshold is encountered, a new cluster is started.
 * @param indices Sequence of index containers, each one containing the indices of a single cluster of points.
 * @return True if the clusters can be retrieved and formed, False otherwise.
 */
inline bool getClusterIndices(const std::vector<Optics::ReachabilityDistance>& reachabilityDistances,
                              const double reachabilityThreshold, std::vector<pcl::PointIndicesPtr>& indices) {
  if (reachabilityDistances.front().reachabilityDistance >= 0.0) {
    return false;
  }
  for (const auto& reachability : reachabilityDistances) {
    if (reachability.reachabilityDistance < 0.0 || reachability.reachabilityDistance >= reachabilityThreshold) {
      pcl::PointIndicesPtr indicesPtr(new pcl::PointIndices);
      indicesPtr->indices.push_back(reachability.pointIndex);
      indices.push_back(indicesPtr);
    } else {
      indices.back()->indices.push_back(reachability.pointIndex);
    }
  }
  return true;
}

/**
 * @brief Compute the density-reachability distances for a given input cloud and output their values into an
 * ordered sequence which can be used to retrieve clusters of density reachable points.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param source Input cloud to compute the reachability distance for.
 * @param minPts Minimal number of point per cluster.
 * @param distances Reachability distances computed from the given input cloud.
 * @param epsilon Radius Neighbors search radius.
 * @return True if the reachability distance have been computed correctly for the entire cloud, False otherwise.
 */
template <typename PointT>
bool computeReachabilityDistances(const typename pcl::PointCloud<PointT>::Ptr& source, const std::size_t minPts,
                                  std::vector<Optics::ReachabilityDistance>& distances, const double epsilon) {
  const auto sourceSize{source->size()};

  if (sourceSize < 2) {
    return false;
  }

  if (epsilon <= 0.0f) {
    std::clog << "Bad epsilon" << std::endl;
    return false;
  }

  std::vector<bool> processed(sourceSize, false);
  std::vector<int> orderedList{};
  orderedList.reserve(sourceSize);
  std::vector<double> reachability(sourceSize, -1.0f);
  std::vector<std::vector<int>> neighbors{};

  {
    pcl::KdTreeFLANN<PointT> kdTree{};
    kdTree.setInputCloud(source);
    std::vector<float> placeHolder{};
    placeHolder.reserve(sourceSize);
    std::for_each(std::begin(*source), std::end(*source),
                  [epsilon, &placeHolder, &kdTree, &neighbors](const PointT& point) {
                    neighbors.emplace_back();
                    kdTree.radiusSearch(point, epsilon, neighbors.back(), placeHolder);
                  });
  }

  if (neighbors.size() != sourceSize) {
    return false;
  }

  for (auto pointIndex{0}; pointIndex < sourceSize; ++pointIndex) {
    if (processed[pointIndex]) {
      continue;
    }
    processed[pointIndex] = true;
    orderedList.push_back(pointIndex);
    std::set<Optics::ReachabilityDistance> seeds{};

    auto& neighborIndices{neighbors[pointIndex]};
    auto coreDistance{std::numeric_limits<double>::max()};
    if (!Optics::internals::computeCoreDistance<PointT>((*source)[pointIndex], source, neighborIndices, minPts,
                                                        coreDistance)) {
      continue;
    }
    Optics::internals::expandCluster((*source)[pointIndex], source, neighborIndices, coreDistance, processed,
                                     reachability, seeds);
    while (!seeds.empty()) {
      const auto& seed{*std::begin(seeds)};
      seeds.erase(std::begin(seeds));
      processed[seed.pointIndex] = true;
      orderedList.push_back(seed.pointIndex);
      const auto& seedNeighborIndices{neighbors[seed.pointIndex]};
      auto seedCoreDistance{std::numeric_limits<double>::max()};
      if (!Optics::internals::computeCoreDistance<PointT>((*source)[seed.pointIndex], source, seedNeighborIndices,
                                                          minPts, seedCoreDistance)) {
        continue;
      }
      Optics::internals::expandCluster((*source)[seed.pointIndex], source, seedNeighborIndices, seedCoreDistance,
                                       processed, reachability, seeds);
    }
  }

  if (orderedList.size() != source->size() || !internals::isUnique(orderedList)) {
    return false;
  }

  distances.clear();
  std::for_each(std::begin(orderedList), std::end(orderedList), [&reachability, &distances](std::size_t point_idx) {
    distances.emplace_back(point_idx, reachability[point_idx]);
  });
  return true;
}

template <typename PointT>
bool computeReachabilityDistances(const typename pcl::PointCloud<PointT>::Ptr& source,
                                  const pcl::IndicesConstPtr& sourceIndices, const std::size_t minPts,
                                  std::vector<Optics::ReachabilityDistance>& distances, const double epsilon) {
  const auto nbPoints{sourceIndices->size()};

  if (nbPoints < 2) {
    return false;
  }

  if (epsilon <= 0.0f) {
    std::clog << "Bad epsilon" << std::endl;
    return false;
  }

  std::unordered_map<int, bool> processed{};
  std::vector<int> orderedList{};
  orderedList.reserve(nbPoints);
  std::unordered_map<int, double> reachabilities{};
  std::unordered_map<int, std::vector<int>> neighbors{};
  reachabilities.reserve(nbPoints);
  neighbors.reserve(nbPoints);
  for (auto index{0u}; index < nbPoints; ++index) {
    processed[(*sourceIndices)[index]] = false;
    reachabilities[(*sourceIndices)[index]] = -1.0f;
  }

  {
    pcl::KdTreeFLANN<PointT> kdTree{};
    kdTree.setInputCloud(source, sourceIndices);
    std::vector<float> placeHolder{};
    placeHolder.reserve(nbPoints);
    std::for_each(std::begin(*sourceIndices), std::end(*sourceIndices),
                  [epsilon, &placeHolder, &kdTree, &neighbors, &source](const int& index) {
                    neighbors[index] = std::vector<int>();
                    kdTree.radiusSearch((*source)[index], epsilon, neighbors[index], placeHolder);
                  });
  }

  if (neighbors.size() != nbPoints) {
    return false;
  }

  for (const auto index : *sourceIndices) {
    if (processed[index]) {
      continue;
    }
    processed[index] = true;
    orderedList.push_back(index);
    std::set<Optics::ReachabilityDistance> seeds{};

    const auto& neighborIndices{neighbors[index]};
    auto coreDistance{std::numeric_limits<double>::max()};
    if (!Optics::internals::computeCoreDistance((*source)[index], source, neighborIndices, minPts, coreDistance)) {
      continue;
    }
    Optics::internals::expandCluster((*source)[index], source, neighborIndices, coreDistance, processed, reachabilities,
                                     seeds);
    while (!seeds.empty()) {
      const auto& seed{*std::begin(seeds)};
      seeds.erase(std::begin(seeds));
      processed[seed.pointIndex] = true;
      orderedList.push_back(seed.pointIndex);
      const auto& seedNeighborIndices{neighbors[seed.pointIndex]};
      auto seedCoreDistance{std::numeric_limits<double>::max()};
      if (!Optics::internals::computeCoreDistance<PointT>((*source)[seed.pointIndex], source, seedNeighborIndices,
                                                          minPts, seedCoreDistance)) {
        continue;
      }
      Optics::internals::expandCluster<PointT>((*source)[seed.pointIndex], source, seedNeighborIndices,
                                               seedCoreDistance, processed, reachabilities, seeds);
    }
  }

  if (orderedList.size() != sourceIndices->size() || !internals::isUnique(orderedList)) {
    return false;
  }

  distances.clear();
  std::for_each(std::begin(orderedList), std::end(orderedList),
                [&reachabilities, &distances](int index) { distances.emplace_back(index, reachabilities[index]); });
  return true;
}

/**
 * @brief Cluster a given input cloud based on the density of its points. The epsilon neighborhood distance parameters
 * is estimated using a heuristic which uses the spatial organization of the input cloud point.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param source Input cloud to make density clusters from.
 * @param minPts Minimal number of point per cluster.
 * @param reachabilityThreshold Maximal reachability distance allowed for a point p to belong to a cluster being
 * formed. If a point with a reachability distance higher than the threshold is encountered, a new cluster is started.
 * @param indices Sequence of index containers, each one containing the indices of a single cluster of points.
 * @return True if the clusters have been correctly generated from the given source, False otherwise.
 */
template <typename PointT>
bool optics(const typename pcl::PointCloud<PointT>::Ptr& source, const std::size_t minPts,
            const double reachabilityThreshold, std::vector<pcl::PointIndicesPtr>& indices) {
  indices.clear();
  indices.reserve(source->size());
  std::vector<Optics::ReachabilityDistance> distances{};
  distances.reserve(source->size());
  if (computeReachabilityDistances<PointT>(source, minPts, distances,
                                           Optics::epsilonEstimation<PointT>(source, minPts))) {
    return getClusterIndices(distances, reachabilityThreshold, indices);
  }
  return false;
}

/**
 * @brief Cluster a given input cloud based on the density of its points. The epsilon neighborhood distance parameters
 * is estimated using a heuristic which uses the spatial organization of the input cloud point.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param source Input cloud to make density clusters from.
 * @param sourceIndices Indices of the point to cluster from the source cloud.
 * @param minPts Minimal number of point per cluster.
 * @param reachabilityThreshold Maximal reachability distance allowed for a point p to belong to a cluster being
 * formed. If a point with a reachability distance higher than the threshold is encountered, a new cluster is started.
 * @param indices Sequence of index containers, each one containing the indices of a single cluster of points.
 * @return True if the clusters have been correctly generated from the given source, False otherwise.
 */
template <typename PointT>
bool optics(const typename pcl::PointCloud<PointT>::Ptr& source, const pcl::IndicesConstPtr& sourceIndices,
            const std::size_t minPts, const double reachabilityThreshold, std::vector<pcl::PointIndicesPtr>& indices) {
  indices.clear();
  indices.reserve(source->size());
  std::vector<Optics::ReachabilityDistance> distances{};
  distances.reserve(source->size());
  if (computeReachabilityDistances<PointT>(source, sourceIndices, minPts, distances,
                                           Optics::epsilonEstimation<PointT>(source, minPts))) {
    return getClusterIndices(distances, reachabilityThreshold, indices);
  }
  return false;
}

/**
 * @brief Cluster a given input cloud based on the density of its points.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param source Input cloud to make density clusters from.
 * @param epsilon Radius Neighbors search radius.
 * @param minPts Minimal number of point per cluster.
 * @param reachabilityThreshold Maximal reachability distance allowed for a point p to belong to a cluster being
 * formed. If a point with a reachability distance higher than the threshold is encountered, a new cluster is started.
 * @param indices Sequence of index containers, each one containing the indices of a single cluster of points.
 * @return True if the clusters have been correctly generated from the given source, False otherwise.
 */
template <typename PointT>
bool optics(const typename pcl::PointCloud<PointT>::Ptr& source, const double epsilon, const std::size_t minPts,
            const double reachabilityThreshold, std::vector<pcl::PointIndicesPtr>& indices) {
  indices.clear();
  indices.reserve(source->size());
  std::vector<Optics::ReachabilityDistance> distances{};
  distances.reserve(source->size());
  if (computeReachabilityDistances<PointT>(source, minPts, distances, epsilon)) {
    return getClusterIndices(distances, reachabilityThreshold, indices);
  }
  return false;
}

/**
 * @brief Cluster a given input cloud based on the density of its points.
 * @tparam PointT Type of pcl points of the cloud to make clusters from.
 * @param source Input cloud to make density clusters from.
 * @param sourceIndices Indices of the point to cluster from the source cloud.
 * @param epsilon Radius Neighbors search radius.
 * @param minPts Minimal number of point per cluster.
 * @param reachabilityThreshold Maximal reachability distance allowed for a point p to belong to a cluster being
 * formed. If a point with a reachability distance higher than the threshold is encountered, a new cluster is started.
 * @param indices Sequence of index containers, each one containing the indices of a single cluster of points.
 * @return True if the clusters have been correctly generated from the given source, False otherwise.
 */
template <typename PointT>
bool optics(const typename pcl::PointCloud<PointT>::Ptr& source, const pcl::IndicesConstPtr& sourceIndices,
            const double epsilon, const std::size_t minPts, const double reachabilityThreshold,
            std::vector<pcl::PointIndicesPtr>& indices) {
  indices.clear();
  indices.reserve(source->size());
  std::vector<Optics::ReachabilityDistance> distances{};
  distances.reserve(source->size());
  if (computeReachabilityDistances<PointT>(source, sourceIndices, minPts, distances, epsilon)) {
    return getClusterIndices(distances, reachabilityThreshold, indices);
  }
  return false;
}

}  // namespace Optics

#endif  // OPTICS_HPP