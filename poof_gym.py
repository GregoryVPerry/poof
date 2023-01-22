"""
Use a variable-sized ES-HyperNEAT network to perform a factoring task
Fitness threshold set in config
- by default very high to show the high possible accuracy of this library.
"""

import neat
import neat.nn
import pickle
import random
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork

PRIMES_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
               31, 37, 41, 43, 47, 53, 59, 61, 67,
               71, 73, 79, 83, 89, 97, 101, 103,
               107, 109, 113, 127, 131, 137, 139,
               149, 151, 157, 163, 167, 173, 179,
               181, 191, 193, 197, 199, 211, 223,
               227, 229, 233, 239, 241, 251, 257,
               263, 269, 271, 277, 281, 283, 293,
               307, 311, 313, 317, 331, 337, 347, 349]

def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {
        "initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
        "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
        "variance_threshold": 0.03,
        "band_threshold": 0.3,
        "iteration_level": 1,
        "division_threshold": 0.5,
        "max_weight": 5.0,
        "activation": "sigmoid",
    }

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

DYNAMIC_PARAMS = params(VERSION)

def eval_fitness(genomes, config):
    """
    Fitness function.
    For each genome evaluate its fitness, in this case, as the mean squared error.
    """

    for _, genome in genomes:

        # generate two random primes for target modulus
        N = 16  # 16-bit factors, don't forget to change binary precision in -> list

        while True:
            p = getLowLevelPrime(N)
            if not isMillerRabinPassed(p):
                continue
            else:
                break

        while True:
            q = getLowLevelPrime(N)
            if not isMillerRabinPassed(q):
                continue
            else:
                break

        target_modulus = p * q

        # create target modulus to be factored
        MODULUS_OUTPUTS = (int_to_binary_tuple_list(target_modulus))

        print(format(p, '#018b') + format(q, '#018b'))
        print(MODULUS_OUTPUTS)

        # generate first prime candidate with features for network inputs
        while True:
            p = getLowLevelPrimeModulo(N, (target_modulus % 9))
            if not isMillerRabinPassed(p):
                continue
            else:
                break

        print('prime candidate p: ' + str(p))
        print('target_modulus: ' + str(target_modulus) + ' % 9 == ' + str(target_modulus & 9))

        while True:
            q = getSecondLowLevelPrimeModulo(N, str(p % 9))
            if not isMillerRabinPassed(q):
                continue
            else:
                break

        FACTOR_INPUTS = (int_to_binary_tuple_list(p) + \
                         int_to_binary_tuple_list(p % 3) + \
                         int_to_binary_tuple_list(p % 4) + \
                         int_to_binary_tuple_list(p % 5) + \
                         int_to_binary_tuple_list(p % 6) + \
                         int_to_binary_tuple_list(p % 7) + \
                         int_to_binary_tuple_list(p % 8) + \
                         int_to_binary_tuple_list(p % 9) + \
                         int_to_binary_tuple_list(sum(int(digit) for digit in str(p))) + \
                         int_to_binary_tuple_list(sum(int(digit) for digit in str(sum(int(digit) for digit in str(p))))) + \
                         int_to_binary_tuple_list(q) + \
                         int_to_binary_tuple_list(q % 3) + \
                         int_to_binary_tuple_list(q % 4) + \
                         int_to_binary_tuple_list(q % 5) + \
                         int_to_binary_tuple_list(q % 6) + \
                         int_to_binary_tuple_list(q % 7) + \
                         int_to_binary_tuple_list(q % 8) + \
                         int_to_binary_tuple_list(q % 9) + \
                         int_to_binary_tuple_list(sum(int(digit) for digit in str(q))) + \
                         int_to_binary_tuple_list(sum(int(digit) for digit in str(sum(int(digit) for digit in str(q))))))

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(SUBSTRATE, cppn, DYNAMIC_PARAMS)
        net = network.create_phenotype_network()

        sum_square_error = 0.0

        for factor_inputs, modulus_expected in zip(FACTOR_INPUTS, MODULUS_OUTPUTS):
            new_factor_input = factor_inputs + (random.random(),)
            net.reset()

            for _ in range(network.activations):
                modulus_output = net.activate(new_factor_input)

            sum_square_error += ((modulus_output[0] - modulus_expected[0]) ** 2.0) / 4.0
            print('modulus_output[0]: ' + str(modulus_output[0]) + ' modulus_expected[0]: ' + str(modulus_expected[0]))
            genome.fitness = 1 - sum_square_error


def run(gens, version):
    """
    Create the population and run the modulus task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """
    pop = neat.population.Population(CONFIG)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    global DYNAMIC_PARAMS
    DYNAMIC_PARAMS = params(version)

    winner = pop.run(eval_fitness, gens)
    print(f"es_hyperneat_factoring_{VERSION_TEXT} done")
    return winner, stats

def int_to_binary_tuple_list(n: int) -> list:
    # Convert the integer to a binary string.
    binary_str = format(n, '#018b')[2:]  # don't forget to add 2 to make length 16

    # Initialize an empty list.
    tuple_list = []

    # Iterate over the binary string.
    for char in binary_str:
        # Append a tuple with (0.0, 0.0) or (1.0, 1.0) depending on the value of the current character.
        if char == '0':
            tuple_list.append((0.0, 0.0))
        else:
            tuple_list.append((1.0, 1.0))

    return tuple_list

def nBitRandom(n):
    return random.randrange(2**(n-1)+1, 2**n - 1)

def getLowLevelPrime(n):
    '''Generate a prime candidate divisible by first primes'''
    while True:
        # Obtain a random number
        pc = nBitRandom(n)

        # Test divisibility by pre-generated primes
        for divisor in PRIMES_LIST:
            if pc % divisor == 0 and divisor**2 <= pc:
                break
        else:
            return pc

def getLowLevelPrimeModulo(n, mod):
    '''Generate a prime candidate divisible by first primes and with root modulo property'''
    while True:
        # Obtain a random number
        pc = nBitRandom(n)

        # Test divisibility by pre-generated primes
        for divisor in PRIMES_LIST:
            if pc % divisor == 0 and divisor**2 <= pc:
                break
        else:
            if (pc % 1) in [1, 2, 4, 5, 6, 8]:
                return pc
            elif (pc % 2) in [1, 2, 4, 5, 7, 8]:
                return pc
            elif (pc % 3) in [1, 2, 3, 4, 5, 6, 7, 8]:
                return pc
            elif (pc % 4) in [1, 2, 4, 5, 7, 8]:
                return pc
            elif (pc % 5) in [1, 2, 4, 5, 7, 8]:
                return pc
            elif (pc % 6) in [1, 2, 3, 4, 5, 6, 7, 8]:
                return pc
            elif (pc % 7) in [1, 2, 4, 5, 7, 8]:
                return pc
            elif (pc % 8) in [1, 2, 4, 5, 7, 8]:
                return pc
            else:
                break

def getLowLevelSecondPrimeModulo(n, mod):
    '''Generate a prime candidate divisible by first primes and with root modulo property'''
    while True:
        # Obtain a random number
        pc = nBitRandom(n)

        # Test divisibility by pre-generated primes
        for divisor in PRIMES_LIST:
            if pc % divisor == 0 and divisor**2 <= pc:
                break
        else:
            if (pc % 1) in [1, 2, 4, 5, 6, 8]:
                return pc
            elif (pc % 2) in [1, 2, 4, 5, 7, 8]:
                return pc
            elif (pc % 3) in [1, 2, 3, 4, 5, 6, 7, 8]:
                return pc
            elif (pc % 4) in [1, 2, 4, 5, 7, 8]:
                return pc
            elif (pc % 5) in [1, 2, 4, 5, 7, 8]:
                return pc
            elif (pc % 6) in [1, 2, 3, 4, 5, 6, 7, 8]:
                return pc
            elif (pc % 7) in [1, 2, 4, 5, 7, 8]:
                return pc
            elif (pc % 8) in [1, 2, 4, 5, 7, 8]:
                return pc
            else:
                break


def isMillerRabinPassed(mrc):
    '''Run 20 iterations of Rabin Miller Primality test'''
    maxDivisionsByTwo = 0
    ec = mrc-1
    while ec % 2 == 0:
        ec >>= 1
        maxDivisionsByTwo += 1
    assert(2**maxDivisionsByTwo * ec == mrc-1)

    def trialComposite(round_tester):
        if pow(round_tester, ec, mrc) == 1:
            return False
        for i in range(maxDivisionsByTwo):
            if pow(round_tester, 2**i * ec, mrc) == mrc-1:
                return False
    return True

    # Set number of trials here
    numberOfRabinTrials = 20
    for i in range(numberOfRabinTrials):
        round_tester = random.randrange(2, mrc)
        if trialComposite(round_tester):
            return False
    return True

if __name__ == "__main__":

    # Network coordinates and the resulting substrate.
    INPUT_COORDINATES = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]
    OUTPUT_COORDINATES = [(0.0, 1.0)]
    SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

    # Config for CPPN.
    CONFIG = neat.config.Config(
        neat.genome.DefaultGenome,
        neat.reproduction.DefaultReproduction,
        neat.species.DefaultSpeciesSet,
        neat.stagnation.DefaultStagnation,
        "config_cppn_factoring"
    )

    WINNER = run(300, VERSION)[0]  # Only relevant to look at the winner.
    print("\nBest genome:\n{!s}".format(WINNER))

    # Verify network output against training data.
    print("\nOutput:")
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NETWORK = ESNetwork(SUBSTRATE, CPPN, DYNAMIC_PARAMS)
    # This will also draw winner_net.
    WINNER_NET = NETWORK.create_phenotype_network(
        filename=f"es_hyperneat_factoring_{VERSION_TEXT}_winner.png"
    )

    for inputs, expected in zip(FACTOR_INPUTS, MODULUS_OUTPUTS):
        new_input = inputs + (1.0,)
        WINNER_NET.reset()

        for i in range(NETWORK.activations):
            output = WINNER_NET.activate(new_input)

        print(
            "  input {!r}, expected output {!r}, got {!r}".format(
                inputs, expected, output
            )
        )

    # Save CPPN if wished reused and draw it to file.
    draw_net(
        CPPN, filename=f"es_hyperneat_factoring_{VERSION_TEXT}_cppn"
    )
    with open(
        f"es_hyperneat_factoring_{VERSION_TEXT}_cppn.pkl", "wb"
    ) as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
