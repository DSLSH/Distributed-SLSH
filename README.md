Introduction
-------------

The following is a pre-release of the code for the Distributed Stratified Locality Sensitive Hashing (DSLSH) project.
Python3 and numpy are required in order to execute the code.
The code-base will potentially undergo substantial changes in the future.


Content
------------

The code is divided into two main modules: `middleware` and `worker_node` in the folder `distributed_SLSH`.
As the name somewhat imply, the former contains the code for the Orchestrator and the latter
the code for a single node. Our implementation of the SLSH algorithm is contained in the module
`worker_node/SLSH`.


How to run
------------

The following instructions assume the user is working from folder `distributed_SLSH`.

*Preliminaries:*
run `python setup.py install` to install prerequisites.

For the user's convenience, we provide a `sample_main.py` file containing trivial examples of how
both the Orchestrator and the nodes are meant to be used and exposing their interfaces.

*Sample script:*
By running `run_sample_main.sh` (in the root folder), the user can execute both a node and the Orchestrator
running the aforementioned examples on the local machine. Please keep on reading for the examples' description
and for instructions on their manual execution/generalization.

**IMPORTANT:**
The execution of Orchestrator and nodes are tighly coupled: all the nodes must be run before the Orchestrator
(otherwise an error will occur).

`test_node` and `test_orchestrator` implement simple examples which employ the unitary matrix
as dataset and adds some noise to form the queries. A simple architecture with one node
and the Orchestrator is assumed. In order to run these functions on the local machine, run
`sample_main.py` from the `distributed_SLSH` folder with the following format:

    `python3 sample_main.py <role> &`

where `role` can be either `orchestrator` or `node`.

By creating functions analogous to those provided in `sample_main.py`
and complying with the node/Orchestrator interfaces, the user can provide
a dataset of her/his choice along with queries and labels. Moreover, we suggest to
change line 78 in order to be able to execute the nodes on different machines from
the Orchestrator and to possibily include more than one node.


Results logging
----------------

The execution results are stored in the folder `results`, which contains `accuracy-testing.txt` having a line per execution
reporting accuracy, recall, and MCC. Folders `results/intranode` and `results/distributed` will
contain a file per execution with detailed query-by-query results.


Tests
------

Additional tests are provided in folders `middleware/tests` and `worker_node/tests`. Run from `distributed_SLSH`
the following command:

    `python3 -m unittest <module>/tests/<testname>`

In case a distributed test is run, as from the `sample_main.py`, the nodes need to be run before the Orchestrator.
