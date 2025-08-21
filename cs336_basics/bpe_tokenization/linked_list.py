from typing import Generic, TypeVar

T = TypeVar("T")


class Node(Generic[T]):
    def __init__(self, val: T):
        self.val = val
        self.prev: Node[T] | None = None
        self.next: Node[T] | None = None

    def __repr__(self):
        return (
            f"Node({self.val}, prev={id(self.prev) if self.prev else None},"
            f"next={id(self.next) if self.next else None})"
        )
    
    def replace_two(self, new_node: "Node[T]") -> None:
        assert self.next, "Next node is None, cannot replace two nodes"
        new_node.prev = self.prev
        new_node.next = self.next.next
        if self.prev:
            self.prev.next = new_node
        if self.next:
            self.next.prev = new_node

class DoubleLinkedList(Generic[T]):
    def __init__(self):
        self.head: Node[T] | None = None
        self.tail: Node[T] | None = None

    @classmethod
    def from_list(cls, lst: list[T]) -> "DoubleLinkedList[T]":
        dll = cls()
        if not lst:
            return dll
        dll.head = Node(lst[0])
        dll.tail = dll.head
        for item in lst[1:]:
            new_node = Node(item)
            dll.tail.next = new_node
            new_node.prev = dll.tail
            dll.tail = new_node

        return dll

    def __repr__(self):
        nodes = []
        current = self.head
        while current:
            nodes.append(repr(current))
            current = current.next
        return " <-> ".join(nodes)
