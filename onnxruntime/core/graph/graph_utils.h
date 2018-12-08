// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph.h"

namespace onnxruntime {

namespace graph_edit_utils {
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain = kOnnxDomainAlias);

/** Checks if the given node has only constant inputs (initializers). */
bool IsConstantInputsNode(const Graph& graph, const Node& node);

/** Builds a subgraph given a Graph and the indices of the nodes of the Graph that will
    be added to the subgraph. */
Status BuildSubgraph(const Graph& graph,
                     const std::vector<onnxruntime::NodeIndex>& subgraph_nodes,
                     Graph& subgraph);

}  // namespace graph_edit_utils

}  // namespace onnxruntime
