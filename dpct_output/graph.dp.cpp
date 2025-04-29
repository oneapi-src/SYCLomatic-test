// ===------- graph.cu ------------------------------------ *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <vector>

const int blockSize = 256;
const int numBlocks = (10 + blockSize - 1) / blockSize;

void init(float *a) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int id = item_ct1.get_local_id(2) +
           item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (id < 10) {
    a[id] = 1.0f;
  }
}

void incrementA(float *a) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int id = item_ct1.get_local_id(2) +
           item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (id < 10) {
    a[id] += 1.0f;
  }
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  dpct::experimental::command_graph_ptr graph;

  dpct::queue_ptr stream;

  stream = dev_ct1.create_queue();

  float *d_a, h_a[10];

  d_a = sycl::malloc_device<float>(10, q_ct1);

  dpct::experimental::begin_recording(stream);

  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                                             sycl::range<3>(1, 1, blockSize),
                                         sycl::range<3>(1, 1, blockSize)),
                       [=](sycl::nd_item<3> item_ct1) {
                         init(d_a);
                       });

  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                                             sycl::range<3>(1, 1, blockSize),
                                         sycl::range<3>(1, 1, blockSize)),
                       [=](sycl::nd_item<3> item_ct1) {
                         incrementA(d_a);
                       });

  dpct::experimental::end_recording(stream, &graph);
  dpct::experimental::command_graph_exec_ptr execGraph;
  dpct::experimental::instantiate(&execGraph, graph);

  dpct::experimental::launch(execGraph, stream);

  stream->wait(); // Ensure the graph has completed execution

  q_ct1.memcpy(h_a, d_a, 10 * sizeof(float)).wait();

  for (int i = 0; i < 10; i++) {
    if (h_a[i] != 2.0f) {
      printf("Results do not match\n");
      return -1;
    }
  }

  size_t numNodes;
  dpct::experimental::get_nodes(graph, nullptr, &numNodes);
  std::vector<dpct::experimental::node_ptr> nodes(numNodes);
  dpct::experimental::get_nodes(graph, nodes.data(), &numNodes);

  // Get root nodes in the graph
  size_t numRootNodes;
  dpct::experimental::get_root_nodes(graph, nullptr, &numRootNodes);
  std::vector<dpct::experimental::node_ptr> rootNodes(numRootNodes);
  dpct::experimental::get_root_nodes(graph, rootNodes.data(), &numRootNodes);

  if (numNodes != 2 || numRootNodes != 1) {
    printf("Number of nodes or root nodes do not match\n");
    return -1;
  }

  printf("Passed\n");

  dev_ct1.destroy_queue(stream);
  dpct::dpct_free(d_a, q_ct1);
  delete (execGraph);

  return 0;
}
