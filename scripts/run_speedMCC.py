import os
from time import sleep
from run_strongscaling import run_middleware_abp_prediction, run_worker_abp_prediction


def run_lsh_layer(node_ips, port, middleware_ip):
    """
    Tuning for AHE prediction on large datasets.

    :param node_ips: list of ips of the workes (nodes)
    :param port: the port the nodes accept connections on.
    :param middleware_ip: ip of the middleware

    :return: nothing
    """

    d = 30
    workers = 5
    n = 801724
    filename = "MIMICIII-ABP-AHE-lag30m-cond30m.data"
    m_out_list = [100, 125, 150, 175, 200]
    L_out_list = [72, 96, 120]
    m_in = 1
    L_in = 1
    alpha = 1
    k = 10
    cores = 8

    for m_out in m_out_list:
        for L_out in L_out_list:
            # Run all workers for this experiment.
            for j in range(workers):
                run_worker_abp_prediction(j + 1, node_ips[j], port, cores, n,
                                          d, m_out, L_out, m_in, L_in, alpha,
                                          k)
            # Run the middleware and wait for it to terminate.
            sleep(30)
            run_middleware_abp_prediction(middleware_ip, node_ips[:workers],
                                          port, cores, n, d, m_out, L_out,
                                          m_in, L_in, alpha, k, filename)
            sleep(280)

            # Change port to avoid issues.
            port += 10


def run_slsh_layer(node_ips, port, middleware_ip, m_out, L_out):
    """
    Tuning for AHE prediction on large datasets.

    :param node_ips: list of ips of the workes (nodes)
    :param port: the port the nodes accept connections on.
    :param middleware_ip: ip of the middleware

    :return: nothing
    """

    d = 30
    workers = 5
    n = 801724
    filename = "MIMICIII-ABP-AHE-lag30m-cond30m.data"
    m_in_list = [40, 65, 90, 115]
    L_in_list = [20, 60]
    alpha = 0.005
    k = 10
    cores = 8

    for m_in in m_in_list:
        for L_in in L_in_list:
            # Run all workers for this experiment.
            for j in range(workers):
                run_worker_abp_prediction(j + 1, node_ips[j], port, cores, n,
                                          d, m_out, L_out, m_in, L_in, alpha,
                                          k)
            # Run the middleware and wait for it to terminate.
            sleep(30)
            run_middleware_abp_prediction(middleware_ip, node_ips[:workers],
                                          port, cores, n, d, m_out, L_out,
                                          m_in, L_in, alpha, k, filename)
            sleep(280)

            # Change port to avoid issues.
            port += 10


def run_baseline(ip):

    command = "\"cd /home/ubuntu/code/distributed_SLSH; python3 ahe_main.py node --mode local --task exhaustive-accuracy --cores {} --n {} --d {} --k {} --filename {}\""
    ssh = "ssh ubuntu@{} ".format(ip)  # User settings.

    if ip == "local":
        ssh = ""
        command = "cd /home/ubuntu/code/distributed_SLSH; python3 ahe_main.py node --mode local --task exhaustive-accuracy --cores {} --n {} --d {} --k {} --filename {}"

    n = 801724
    d = 30
    filename = "MIMICIII-ABP-AHE-lag30m-cond30m.data"
    cores = 1
    k = 1

    print(command.format(cores, n, d, k, filename))
    os.system(ssh + command.format(cores, n, d, k, filename))


if __name__ == "__main__":
    """
    Usage:

        run from scripts/
        python3 run_speedMCC.py


    IMPORTANT: in order to run a script without keeping the terminal open, use
    nohup python3 -u run_speedMCC.py &
    """

    node_ips = ["128.52.161.48", "128.52.161.31"]  # User settings.
    port = 3000  # The port the nodes accept connections on.  # User settings.
    middleware_ip = "128.52.161.43"  # User settings.

    run_lsh_layer(node_ips, port, middleware_ip)

    m_out = 125
    L_out = 120
    run_slsh_layer(node_ips, port, middleware_ip, m_out, L_out)

    run_baseline(middleware_ip)
