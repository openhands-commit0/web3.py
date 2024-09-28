from collections import OrderedDict
from collections.abc import Hashable
from typing import Any, Callable, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, cast
from eth_utils import is_integer
from web3._utils.formatters import recursive_map
T = TypeVar('T')
TKey = TypeVar('TKey', bound=Hashable)
TValue = TypeVar('TValue')

class ReadableAttributeDict(Mapping[TKey, TValue]):
    """
    The read attributes for the AttributeDict types
    """

    def __init__(self, dictionary: Dict[TKey, TValue], *args: Any, **kwargs: Any) -> None:
        self.__dict__ = dict(dictionary)
        self.__dict__.update(dict(*args, **kwargs))

    def __getitem__(self, key: TKey) -> TValue:
        return self.__dict__[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.__dict__!r})'

    def _repr_pretty_(self, builder: Any, cycle: bool) -> None:
        """
        Custom pretty output for the IPython console
        https://ipython.readthedocs.io/en/stable/api/generated/IPython.lib.pretty.html#extending  # noqa: E501
        """
        pass

class MutableAttributeDict(MutableMapping[TKey, TValue], ReadableAttributeDict[TKey, TValue]):

    def __setitem__(self, key: Any, val: Any) -> None:
        self.__dict__[key] = val

    def __delitem__(self, key: Any) -> None:
        del self.__dict__[key]

class AttributeDict(ReadableAttributeDict[TKey, TValue], Hashable):
    """
    This provides superficial immutability, someone could hack around it
    """

    def __setattr__(self, attr: str, val: TValue) -> None:
        if attr == '__dict__':
            super().__setattr__(attr, val)
        else:
            raise TypeError('This data is immutable -- create a copy instead of modifying')

    def __delattr__(self, key: str) -> None:
        raise TypeError('This data is immutable -- create a copy instead of modifying')

    def __hash__(self) -> int:
        return hash(tuple(sorted(tupleize_lists_nested(self).items())))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttributeDict):
            return hash(self) == hash(other)
        elif isinstance(other, Mapping):
            return self.__dict__ == dict(other)
        else:
            return False

def tupleize_lists_nested(d: Mapping[TKey, TValue]) -> AttributeDict[TKey, TValue]:
    """
    Unhashable types inside dicts will throw an error if attempted to be hashed.
    This method converts lists to tuples, rendering them hashable.
    Other unhashable types found will raise a TypeError
    """
    pass

class NamedElementOnion(Mapping[TKey, TValue]):
    """
    Add layers to an onion-shaped structure. Optionally, inject to a specific layer.
    This structure is iterable, where the outermost layer is first, and innermost
    is last.
    """

    def __init__(self, init_elements: Sequence[Any], valid_element: Callable[..., bool]=callable) -> None:
        self._queue: 'OrderedDict[Any, Any]' = OrderedDict()
        for element in reversed(init_elements):
            if valid_element(element):
                self.add(element)
            else:
                self.add(*element)

    def inject(self, element: TValue, name: Optional[TKey]=None, layer: Optional[int]=None) -> None:
        """
        Inject a named element to an arbitrary layer in the onion.

        The current implementation only supports insertion at the innermost layer,
        or at the outermost layer. Note that inserting to the outermost is equivalent
        to calling :meth:`add` .
        """
        pass

    @property
    def middlewares(self) -> Sequence[Any]:
        """
        Returns middlewares in the appropriate order to be imported into a new Web3
        instance (reversed _queue order) as a list of (middleware, name) tuples.
        """
        pass

    def __iter__(self) -> Iterator[TKey]:
        elements = self._queue.values()
        if not isinstance(elements, Sequence):
            elements = list(elements)
        return iter(reversed(elements))

    def __add__(self, other: Any) -> 'NamedElementOnion[TKey, TValue]':
        if not isinstance(other, NamedElementOnion):
            return NotImplemented
        combined = self._queue.copy()
        combined.update(other._queue)
        return NamedElementOnion(cast(List[Any], combined.items()))

    def __contains__(self, element: Any) -> bool:
        return element in self._queue

    def __getitem__(self, element: TKey) -> TValue:
        return self._queue[element]

    def __len__(self) -> int:
        return len(self._queue)

    def __reversed__(self) -> Iterator[TValue]:
        elements = cast(List[Any], self._queue.values())
        if not isinstance(elements, Sequence):
            elements = list(elements)
        return iter(elements)