// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/initializer.h"
#include "core/graph/conv_activation_fusion.h"

using namespace onnx;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
bool IsFusableActivation(const Node& node) {
  const std::string& op_type = node.OpType();
  return op_type == "LeakyRelu" || op_type == "Relu" || op_type == "Sigmoid" || op_type == "Tanh";
}
}  // namespace

Status ConvActivationFusion::Apply(Graph& graph, bool& modified) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  std::vector<onnxruntime::NodeIndex> removed_nodes;
  for (auto index : order) {
    auto node = graph.GetNode(index);
    if (node->OpType() != "Conv" || node->GetOutputEdgesCount() != 1) {
      continue;
    }
    const Node& next_node = *(node->OutputNodesBegin());
    if (!IsFusableActivation(next_node) || graph.IsNodeOutputsInGraphOutputs(next_node)) {
      continue;
    }

    Node* conv_node = node;
    const Node& act_node = next_node;
    std::vector<NodeArg> input_args, output_args;

    Node& fused_conv = graph.AddNode("fused " + conv_node->Name(), "FusedConv",
                                     "fused Conv " + conv_node->Name() + "with activation " + act_node.OpType(),
                                     conv_node->MutableInputDefs(),
                                     conv_node->MutableOutputDefs(),
                                     &conv_node->GetAttributes(),
                                     "com.microsoft");

    //Add a new attribute to specify the activation type
    AttributeProto act;
    act.set_name("activation");
    act.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    act.set_s(act_node.OpType());
    fused_conv.AddAttribute("activation", act);

    //Add optional attributes for activations
    if (act_node.OpType() == "LeakyRelu") {
      const NodeAttributes attrs = act_node.GetAttributes();
      for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        fused_conv.AddAttribute(it->first, it->second);
      }
    }

    // Replace the input of the node following activation node
    const NodeArg* act_output_def = act_node.OutputDefs()[0];
    NodeArg* fused_conv_output_def = fused_conv.MutableOutputDefs()[0];
    for (auto it = act_node.OutputNodesBegin(); it != act_node.OutputNodesEnd(); ++it) {
      auto output_node = graph.GetNode((*it).Index());
      if (!output_node) {
        return Status(ONNXRUNTIME, INVALID_ARGUMENT);
      }

      auto& input_defs = output_node->MutableInputDefs();
      for (auto& def : input_defs) {
        if (def == act_output_def) {
          def = fused_conv_output_def;
        }
      }
    }

    removed_nodes.push_back(act_node.Index());
    removed_nodes.push_back(conv_node->Index());
  }

  for (auto i : removed_nodes) {
    graph.RemoveNode(i);
  }

  if (!removed_nodes.empty()) {
    modified = true;
    ONNXRUNTIME_RETURN_IF_ERROR(graph.Resolve());
  }
  return Status::OK();
}
}  // namespace onnxruntime
