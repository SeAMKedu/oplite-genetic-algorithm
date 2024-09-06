"""GA application"""
import copy
import json
import socket

import pygad

import config
import genetic
import utils

HOST = config.get("server.host")
PORT = config.get("server.port")
USE_PYGAD = config.get("ga.use_pygad")
POPULATION_SIZE = config.get("ga.population_size")
NUM_GENERATIONS = config.get("ga.num_generations")
ELITE_SIZE = config.get("ga.elite_size")
MUTATION_RATE = config.get("ga.mutation_rate")
TRANSFER_TIMES = config.get("storage.transfer_times")
PICK_TIME = config.get("storage.pick_time")
PLACE_TIME = config.get("storage.place_time")

storage = None
transfers = []


def compute_fitness(ga_instance, individual: list, index: int) -> float:
    """
    Compute the fitness of the individual, i.e. solution candidate.

    :param ga_instance: Instance of the genetic algorithm class.
    :param individual: List of integers.
    :param index: Index of the individual in the population.
    :returns: Fitness of the individual.

    """
    global storage
    global transfers

    # Storage inventory.
    inventory = copy.deepcopy(storage.get_inventory())
    # Each integer in the individual corresponds to a specific transfer.
    transfers_ = [transfers[i] for i in individual]

    # Compute the total duration of the transfers in seconds.
    duration = 0.0
    for i, transfer in enumerate(transfers_):
        src = transfer["src"]  # pick position
        dst = transfer["dst"]  # place position
        # Crane moves from start position to the first pick position.
        if i == 0:
            duration += TRANSFER_TIMES["P00"][src]
        # Empty crane moves from the place position the next pick position.
        if i > 0:
            previous_transfer = transfers_[i - 1]
            previous_place_pos = previous_transfer["dst"]
            next_pick_pos = src
            if not previous_place_pos == next_pick_pos:
                move_time = TRANSFER_TIMES[previous_place_pos][next_pick_pos]
                duration += move_time
        # Transfer time from the pick position to the place position.
        transfer_time = TRANSFER_TIMES[src][dst]
        # Check if the destination position already contains a pallet. If true,
        # add a "penalty" to increase the duration and lower the fitness value.
        sku_on_destination = inventory[dst]["sku"]
        if sku_on_destination in ("epallet", "mpallet"):
            transfer_time = 1000.0  # "penalty"
        else:
            # Update the storage inventory.
            inventory[src]["pid"] = 0
            inventory[src]["sku"] = None
            inventory[dst]["pid"] = transfer["pid"]
            inventory[dst]["sku"] = transfer["sku"]

        duration += PICK_TIME + transfer_time + PLACE_TIME

    fitness = duration
    # pygad package wants to maximize the fitness value. Since the goal is to
    # find the shortest duration, take the inverse of the duration to compute
    # the fitness of the individual.
    if USE_PYGAD:
        fitness = 1 / duration

    return fitness


def main():
    global storage
    global transfers

    # Initialize the storage inventory.
    storage = utils.Storage()

    # TCP server socket.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setblocking(False)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Server socket is listening on {HOST}:{PORT}")

    islistening = True

    while islistening:
        try:
            connection, _ = server.accept()
            connection.setblocking(False)

            # Receive a message from the Visual Components.
            print("----------------------------------------")
            message = connection.recv(4096).decode()
            msgdict = json.loads(message)
            request = msgdict["request"]
            response = {"response": "ok"}

            # Use the first step of the production plan to set the
            # initial state of the storage inventory.
            if request == "setInventory":
                print("Setting storage inventory...")
                data = json.loads(msgdict["data"])
                for value in data["0"].values():
                    storage.add_pallet(value["src"], value["pid"])
                print("done")

            # Use the genetic algorithm (GA) to optimize the transfers.
            elif request == "optimizeTransfers":
                print("GA: Optimizing transfers...")
                transfers = msgdict["data"]
                solution = []
                fitness = 0.0
                num_genes = len(transfers)
                genes = [i for i in range(num_genes)]
                if USE_PYGAD:
                    init_pop = utils.init_population(genes, POPULATION_SIZE)
                    ga = pygad.GA(
                        num_generations=NUM_GENERATIONS,
                        num_parents_mating=num_genes,
                        fitness_func=compute_fitness,
                        sol_per_pop=POPULATION_SIZE,
                        initial_population=init_pop,
                        gene_type=int,
                        num_genes=num_genes,
                        keep_elitism=ELITE_SIZE,
                        mutation_probability=MUTATION_RATE,
                    )
                    ga.run()
                    solution, fitness, index = ga.best_solution()
                    fitness = 1 / fitness
                else:
                    ga = genetic.GeneticAlgorithm(
                        direction=genetic.MINIMUM_DIRECTION,
                        genes=genes,
                        population_size=POPULATION_SIZE,
                        num_generations=NUM_GENERATIONS,
                        elite_size=ELITE_SIZE,
                        mutation_rate=MUTATION_RATE,
                        fitness_function=compute_fitness,
                    )
                    solution, fitness = ga.run()
                # Show solution.
                print(f"GA: Best fitness: {fitness}")
                print("GA: Best solution:")
                solution_transfers = []
                for i in solution:
                    transfer = transfers[i]
                    solution_transfers.append(transfer)
                    print(transfer)
                # Update the storage inventory.
                storage.set_inventory(solution_transfers)
                response = {
                    "fitness": fitness,
                    "solution": solution_transfers,
                }

            # Send a response.
            data = json.dumps(response)
            connection.sendall(data.encode())

        except socket.error:
            pass
        except KeyboardInterrupt:
            islistening = False

    server.close()
    print("<<< Server closed >>>")

if __name__ == "__main__":
    main()
