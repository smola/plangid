from sklearn.tree import _tree


def explain_path(tree, sample):
    if hasattr(tree, "tree_"):
        tree = tree.tree_
    cur_node = 0
    result = []

    while True:
        feature = tree.feature[cur_node]
        if feature == _tree.TREE_UNDEFINED:
            result.append((None, None, tree.value[cur_node].tolist()[0]))
            break
        threshold = tree.threshold[cur_node]
        value = sample[feature]
        if value <= threshold:
            result.append((feature, "lte", threshold))
            cur_node = tree.children_left[cur_node]
        else:
            result.append((feature, "gt", threshold))
            cur_node = tree.children_right[cur_node]
    return result
