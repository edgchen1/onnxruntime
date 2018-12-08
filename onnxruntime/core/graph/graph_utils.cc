
#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace graph_edit_utils {
// fusion is only done for ONNX domain ops
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain) {
  if (node.OpType() != op_type ||
      node.Op()->Deprecated() || node.Op()->SinceVersion() != version ||
      (!node.Domain().empty() && node.Domain() != domain)) {
    return false;
  }
  return true;
}

bool IsConstantInputsNode(const Graph& graph, const Node& node) {
  if (node.GetInputEdgesCount() > 0) {
    return false;
  }
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  for (const auto* input_def : node.InputDefs()) {
    if (!graph.GetInitializedTensor(input_def->Name(), initializer)) {
      return false;
    }
  }
  return true;
}

Status BuildSubgraph(const Graph& graph,
                     const std::vector<onnxruntime::NodeIndex>& subgraph_nodes,
                     Graph& subgraph) {
  std::vector<onnxruntime::NodeIndex> subgraph_nodes;
  subgraph_nodes.push_back(node.Index());

  // Add nodes to subgraph.
  for (auto& node_index : subgraph_nodes) {
    // TODO Can we directly copy the node instead of re-creating it?
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node.InputDefs()) {
      auto& n_input = subgraph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
    }
    for (auto output : node.OutputDefs()) {
      auto& n_output = subgraph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    subgraph.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
  }

  //TODO Check if we don't need to resolve the graph.
  ONNXRUNTIME_ENFORCE(subgraph.Resolve().IsOK());

  // TODO Add initializers?

	return Status::OK();
}


}  // namespace graph_edit_utils

}  // namespace onnxruntime
