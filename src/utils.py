import copy
import random


class Storage:
    """Class that represents the storage."""

    def __init__(self) -> None:
        self.inventory = self._initialize_inventory()

    def _initialize_inventory(self) -> dict:
        inventory = {}
        position_count = 27
        # PID = part ID, SKU = stock keeping unit.
        position_inventory = {"pid": 0, "sku": None}
        for position in range(1, position_count + 1):
            pos = f"P{str(position).zfill(2)}"  # P01, P02, ..., P27
            inventory[pos] = copy.deepcopy(position_inventory)
        return inventory

    def add_part(self, position: str, pid: int, sku: str = "epallet"):
        """Add a part to the given position in the storage."""
        self.inventory[position]["pid"] = pid
        self.inventory[position]["sku"] = sku

    def del_part(self, position: str):
        """Delete a part from the given position in the storage."""
        self.inventory[position]["pid"] = 0
        self.inventory[position]["sku"] = None

    def get_inventory(self) -> dict:
        """Return the storage inventory."""
        return self.inventory

    def set_inventory(self, transfers: list[dict]):
        """Update the storage inventory."""
        for transfer in transfers:
            src = transfer["src"]  # source position
            dst = transfer["dst"]  # destination position
            pid = transfer["pid"]  # part ID
            sku = transfer["sku"]  # stock keeping unit
            self.del_part(src)
            self.add_part(dst, pid, sku)


def init_population(genes: list[int], population_size: int) -> list:
    """
    Initialize the population to be used in the GA computation.

    :param genes: List of integers.
    :param population_size: Size of the population.
    :returns: Population for the GA.

    """
    gene_count = len(genes)
    population = []
    for _ in range(population_size):
        individual = random.sample(genes, gene_count)
        population.append(individual)
    return population
