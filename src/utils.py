import copy
import random


class Storage:
    """High bay rack storage."""

    def __init__(self) -> None:
        self.num_positions = 27
        self.inventory = self._init_inventory()

    def _init_inventory(self) -> dict:
        inventory = {}
        # PID = pallet ID, SKU = stock keeping unit.
        data = {"pid": 0, "sku": None}
        for position in range(1, self.num_positions + 1):
            pos = f"P{str(position).zfill(2)}"  # P01, P02, ..., P27
            inventory[pos] = copy.deepcopy(data)
        return inventory

    def add_pallet(self, position: str, pid: int, sku: str = "epallet"):
        """Add a pallet to the given position in the storage."""
        self.inventory[position]["pid"] = pid
        self.inventory[position]["sku"] = sku

    def remove_pallet(self, position: str):
        """Remove a pallet from the given position in the storage."""
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
            pid = transfer["pid"]  # pallet ID
            sku = transfer["sku"]  # stock keeping unit
            self.remove_pallet(src)
            self.add_pallet(dst, pid, sku)


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
