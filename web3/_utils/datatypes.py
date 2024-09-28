from typing import Any, Collection, Dict, Optional, Tuple, Type
from eth_utils import apply_formatters_to_dict
from eth_utils.toolz import concat

class PropertyCheckingFactory(type):

    def __init__(cls, name: str, bases: Tuple[Type[Any], ...], namespace: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
        super().__init__(name, bases, namespace)

    def __new__(mcs, name: str, bases: Tuple[type], namespace: Dict[str, Any], normalizers: Optional[Dict[str, Any]]=None) -> 'PropertyCheckingFactory':
        all_bases = set(concat((base.__mro__ for base in bases)))
        all_keys = set(concat((base.__dict__.keys() for base in all_bases)))
        for key in namespace:
            verify_attr(name, key, all_keys)
        if normalizers:
            processed_namespace = apply_formatters_to_dict(normalizers, namespace)
        else:
            processed_namespace = namespace
        return super().__new__(mcs, name, bases, processed_namespace)