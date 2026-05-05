#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import json
import os
from typing import Optional

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.multibatching.problem import (
    Location,
    MultibatchingProblem,
    Product,
    TransportLink,
    TransportType,
)


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for multibatching.

    Args:
        data_folder: folder where datasets for multibatching should be found.
            If None, we look in "multibatching" subdirectory of `data_home`.
        data_home: root directory for all datasets. If None, set by
            default to "~/discrete_optimization_data"

    Returns:
        List of absolute paths to JSON files in the data folder.

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/multibatching"

    try:
        files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def parse_json_to_problem(
    json_data: dict,
    scale_capacity: float = 1.0,
    scale_size: float = 1.0,
    scale_co2: float = 1.0,
) -> MultibatchingProblem:
    """Parse a JSON dictionary into a MultibatchingProblem object.

    Args:
        json_data: Dictionary with keys:
            - "transportResources": dict mapping transport IDs to transport data
            - "products": dict mapping product IDs to product data
            - "locations": dict mapping location IDs to location data
            - "routes": dict mapping route IDs to route data
            - "settings": dict with problem settings
        scale_capacity: scaling factor for transport capacities (default: 1.0)
        scale_size: scaling factor for product sizes (default: 1.0)
        scale_co2: scaling factor for CO2 emissions (default: 1.0)

    Returns:
        MultibatchingProblem instance.

    """
    # 1. Parse Transport Types (from transportResources)
    transport_types_map = {
        _id: TransportType(
            id=_id,
            name=data["name"],
            cost=data["cost"],
            speed=data["speed"],
            emissions=data["co2Emissions"] * scale_co2,
            capacity=int(data["capacity"] * scale_capacity),
        )
        for _id, data in json_data["transportResources"].items()
    }

    # 2. Parse Products
    products_map = {
        _id: Product(
            id=_id,
            name=data["name"],
            size=int(data["size"] * scale_size),
            value=data["value"],
            valid_transports=frozenset(
                transport_types_map[tr_id] for tr_id in data["validTR"]
            ),
        )
        for _id, data in json_data["products"].items()
    }

    # 3. Parse Locations
    locations_map = {
        _id: Location(id=_id, name=data["name"], net_supply={})
        for _id, data in json_data["locations"].items()
    }

    # 4. Update locations with net supply/demand from products
    for product_key, product_data in json_data["products"].items():
        product = products_map[product_key]
        for location_key, quantity in product_data["netSupplyDemand"].items():
            if location_key in locations_map:
                locations_map[location_key].net_supply[product] = quantity

    locations_list = list(locations_map.values())

    # 5. Parse Transport Links (routes)
    transport_links_list = []
    for route_id, route_data in json_data["routes"].items():
        loc1 = locations_map[route_data["from"]]
        loc2 = locations_map[route_data["to"]]

        for tr_id, tr_data in route_data["transportResources"].items():
            transport_type = transport_types_map[tr_id]
            distance = tr_data["distance"]

            transport_links_list.append(
                TransportLink(
                    id=f"{route_id}_tr_{tr_id}",
                    location_l1=loc1,
                    location_l2=loc2,
                    distance=distance,
                    transport_type=transport_type,
                )
            )

    # 6. Create the final problem instance
    return MultibatchingProblem(
        transport_types=list(transport_types_map.values()),
        products=list(products_map.values()),
        locations=locations_list,
        transport_links=transport_links_list,
    )


def parse_file(
    file_path: str,
    scale_capacity: float = 1.0,
    scale_size: float = 1.0,
    scale_co2: float = 1.0,
) -> MultibatchingProblem:
    """Parse a JSON file into a MultibatchingProblem.

    Args:
        file_path: Path to the JSON file.
        scale_capacity: scaling factor for transport capacities (default: 1.0)
        scale_size: scaling factor for product sizes (default: 1.0)
        scale_co2: scaling factor for CO2 emissions (default: 1.0)

    Returns:
        MultibatchingProblem instance.

    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return parse_json_to_problem(
        json_data,
        scale_capacity=scale_capacity,
        scale_size=scale_size,
        scale_co2=scale_co2,
    )
