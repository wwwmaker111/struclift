import tree_sitter_c as tsc
from tree_sitter import Parser, Language

lang = Language(tsc.language())
parser = Parser(lang)
tree = parser.parse(b'int main() { return 0; }')
print('root:', tree.root_node.type)
print('children:', [c.type for c in tree.root_node.children])
print('OK')
