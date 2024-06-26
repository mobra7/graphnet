from abc import ABC, abstractmethod
from copy import deepcopy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,2,1,0"
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type,
                    Union, cast)
from tqdm import tqdm

import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import math
import numpy as np
import pandas as pd
import random
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.adam import Adam
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR, GRAPHNET_ROOT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import Dataset, ParquetDataset, SQLiteDataset
from graphnet.data.dataset.dataset import ColumnMissingException, EnsembleDataset, parse_graph_definition
from graphnet.data.utilities.string_selection_resolver import (
    StringSelectionResolver
    )
from graphnet.models import Model, StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.gnn.gnn import GNN
from graphnet.models.graphs import GraphDefinition, KNNGraph
from graphnet.models.task import StandardLearnedTask
from graphnet.models.task.classification import freedom_BinaryClassificationTask
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.labels import Label
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.training.utils import collate_fn
from graphnet.utilities.config import (Configurable, DatasetConfig,
                                       DatasetConfigSaverABCMeta)
from graphnet.utilities.logging import Logger

class freedom_Dataset(
    Logger,
    Configurable,
    torch.utils.data.Dataset,
    ABC,
    metaclass=DatasetConfigSaverABCMeta,
):
    """Base Dataset class for reading from any intermediate file format."""

    # Class method(s)
    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        source: Union[DatasetConfig, str],
    ) -> Union[
        "Dataset",
        "EnsembleDataset",
        Dict[str, "Dataset"],
        Dict[str, "EnsembleDataset"],
    ]:
        """Construct `Dataset` instance from `source` configuration."""
        if isinstance(source, str):
            source = DatasetConfig.load(source)

        assert isinstance(source, DatasetConfig), (
            f"Argument `source` of type ({type(source)}) is not a "
            "`DatasetConfig`"
        )

        assert (
            "graph_definition" in source.dict().keys()
        ), "`DatasetConfig` incompatible with current GraphNeT version."

        # Parse set of `selection``.
        if isinstance(source.selection, dict):
            return cls._construct_datasets_from_dict(source)
        elif (
            isinstance(source.selection, list)
            and len(source.selection)
            and isinstance(source.selection[0], str)
        ):
            return cls._construct_dataset_from_list_of_strings(source)

        cfg = source.dict()
        if cfg["graph_definition"] is not None:
            cfg["graph_definition"] = parse_graph_definition(cfg)
        return source._dataset_class(**cfg)

    @classmethod
    def concatenate(
        cls,
        datasets: List["Dataset"],
    ) -> "EnsembleDataset":
        """Concatenate multiple `Dataset`s into one instance."""
        return EnsembleDataset(datasets)

    @classmethod
    def _construct_datasets_from_dict(
        cls, config: DatasetConfig
    ) -> Dict[str, "Dataset"]:
        """Construct `Dataset` for each entry in dict `self.selection`."""
        assert isinstance(config.selection, dict)
        datasets: Dict[str, "Dataset"] = {}
        selections: Dict[str, Union[str, List]] = deepcopy(config.selection)
        for key, selection in selections.items():
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, (Dataset, EnsembleDataset))
            datasets[key] = dataset

        # Reset `selections`.
        config.selection = selections

        return datasets

    @classmethod
    def _construct_dataset_from_list_of_strings(
        cls, config: DatasetConfig
    ) -> "Dataset":
        """Construct `Dataset` for each entry in list `self.selection`."""
        assert isinstance(config.selection, list)
        datasets: List["Dataset"] = []
        selections: List[str] = deepcopy(cast(List[str], config.selection))
        for selection in selections:
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, Dataset)
            datasets.append(dataset)

        # Reset `selections`.
        config.selection = selections

        return cls.concatenate(datasets)

    @classmethod
    def _resolve_graphnet_paths(
        cls, path: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        if isinstance(path, list):
            return [cast(str, cls._resolve_graphnet_paths(p)) for p in path]

        assert isinstance(path, str)
        return (
            path.replace("$graphnet", GRAPHNET_ROOT_DIR)
            .replace("$GRAPHNET", GRAPHNET_ROOT_DIR)
            .replace("${graphnet}", GRAPHNET_ROOT_DIR)
            .replace("${GRAPHNET}", GRAPHNET_ROOT_DIR)
        )

    def __init__(
        self,
        path: Union[str, List[str]],
        graph_definition: GraphDefinition,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Construct Dataset.

        Args:
            path: Path to the file(s) from which this `Dataset` should read.
            pulsemaps: Name(s) of the pulse map series that should be used to
                construct the nodes on the individual graph objects, and their
                features. Multiple pulse series maps can be used, e.g., when
                different DOM types are stored in different maps.
            features: List of columns in the input files that should be used as
                node features on the graph objects.
            truth: List of event-level columns in the input files that should
                be used added as attributes on the  graph objects.
            node_truth: List of node-level columns in the input files that
                should be used added as attributes on the graph objects.
            index_column: Name of the column in the input files that contains
                unique indicies to identify and map events across tables.
            truth_table: Name of the table containing event-level truth
                information.
            node_truth_table: Name of the table containing node-level truth
                information.
            string_selection: Subset of strings for which data should be read
                and used to construct graph objects. Defaults to None, meaning
                all strings for which data exists are used.
            selection: The events that should be read. This can be given either
                as list of indicies (in `index_column`); or a string-based
                selection used to query the `Dataset` for events passing the
                selection. Defaults to None, meaning that all events in the
                input files are read.
            dtype: Type of the feature tensor on the graph obadditionallyjects returned.
            loss_weight_table: Name of the table containing per-event loss
                weights.
            loss_weight_column: Name of the column in `loss_weight_table`
                containing per-event loss weights. This is also the name of the
                corresponding attribute assigned to the graph object.
            loss_weight_default_value: Default per-event loss weight.
                NOTE: This default value is only applied when
                `loss_weight_table` and `loss_weight_column` are specified, and
                in this case to events with no value in the corresponding
                table/column. That is, if no per-event loss weight table/column
                is provided, this value is ignored. Defaults to None.
            seed: Random number generator seed, used for selecting a random
                subset of events when resolving a string-based selection (e.g.,
                `"10000 random events ~ event_no % 5 > 0"` or `"20% random
                events ~ event_no % 5 > 0"`).
            graph_definition: Method that defines the graph representation.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        # Resolve reference to `$GRAPHNET` in path(s)
        path = self._resolve_graphnet_paths(path)

        # Member variable(s)
        self._path = path
        self._selection = None
        self._pulsemaps = pulsemaps
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._loss_weight_default_value = loss_weight_default_value
        self._graph_definition = deepcopy(graph_definition)

        if node_truth is not None:
            assert isinstance(node_truth_table, str)
            if isinstance(node_truth, str):
                node_truth = [node_truth]

        self._node_truth = node_truth
        self._node_truth_table = node_truth_table

        if string_selection is not None:
            self.warning(
                (
                    "String selection detected.\n "
                    f"Accepted strings: {string_selection}\n "
                    "All other strings are ignored!"
                )
            )
            if isinstance(string_selection, int):
                string_selection = [string_selection]

        self._string_selection = string_selection

        self._selection = None
        if self._string_selection:
            self._selection = f"string in {str(tuple(self._string_selection))}"

        self._loss_weight_column = loss_weight_column
        self._loss_weight_table = loss_weight_table
        if (self._loss_weight_table is None) and (
            self._loss_weight_column is not None
        ):
            self.warning("Error: no loss weight table specified")
            assert isinstance(self._loss_weight_table, str)
        if (self._loss_weight_table is not None) and (
            self._loss_weight_column is None
        ):
            self.warning("Error: no loss weight column specified")
            assert isinstance(self._loss_weight_column, str)

        self._dtype = dtype

        self._label_fns: Dict[str, Callable[[Data], Any]] = {}

        self._string_selection_resolver = StringSelectionResolver(
            self,
            index_column=index_column,
            seed=seed,
        )

        # Implementation-specific initialisation.
        self._init()

        # Set unique indices
        self._indices: Union[List[tuple], List[List[tuple]]]
        if selection is None:
            self._indices = self._get_all_indices()
        elif isinstance(selection, str):
            indices = self._resolve_string_selection_to_indices(
                selection
            )
            self._indices = [(num, 0) for num in indices] + [(num, 1) for num in indices]
        else:
            self._indices = selection

        assert isinstance(self._indices[0],tuple)

        # Purely internal member variables
        self._missing_variables: Dict[str, List[str]] = {}
        self._remove_missing_columns()

        # Implementation-specific post-init code.
        self._post_init()

    # Properties
    @property
    def path(self) -> Union[str, List[str]]:
        """Path to the file(s) from which this `Dataset` reads."""
        return self._path

    @property
    def truth_table(self) -> str:
        """Name of the table containing event-level truth information."""
        return self._truth_table

    # Abstract method(s)
    @abstractmethod
    def _init(self) -> None:
        """Set internal representation needed to read data from input file."""

    def _post_init(self) -> None:
        """Implementation-specific code executed after the main constructor."""

    @abstractmethod
    def _get_all_indices(self) -> List[tuple]:
        """Return a list of all unique values in `self._index_column`."""

    @abstractmethod
    def _get_event_index(
        self, sequential_index: Optional[int]
    ) -> Optional[int]:
        """Return the event index corresponding to a `sequential_index`."""

    @abstractmethod
    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table: Table to be queried.
            columns: Columns to read out.
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`. If no value
                is provided, the entire column is returned.
            selection: Selection to be imposed before reading out data.
                Defaults to None.

        Returns:
            List of tuples containing the values in `columns`. If the `table`
                contains only scalar data for `columns`, a list of length 1 is
                returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """

    # Public method(s)
    def add_label(
        self, fn: Callable[[Data], Any], key: Optional[str] = None
    ) -> None:
        """Add custom graph label define using function `fn`."""
        if isinstance(fn, Label):
            key = fn.key
        assert isinstance(
            key, str
        ), "Please specify a key for the custom label to be added."
        assert (
            key not in self._label_fns
        ), f"A custom label {key} has already been defined."
        self._label_fns[key] = fn

    def __len__(self) -> int:
        """Return number of graphs in `Dataset`."""
        return len(self._indices)

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth, loss_weight = self._query(
            sequential_index
        )
        graph = self._create_graph(features, truth, node_truth, loss_weight)
        return graph

    # Internal method(s)
    def _resolve_string_selection_to_indices(
        self, selection: str
    ) -> List[int]:
        """Resolve selection as string to list of indices.

        Selections are expected to have pandas.DataFrame.query-compatible
        syntax, e.g., ``` "event_no % 5 > 0" ``` Selections may also specify a
        fixed number of events to randomly sample, e.g., ``` "10000 random
        events ~ event_no % 5 > 0" "20% random events ~ event_no % 5 > 0" ```
        """
        return self._string_selection_resolver.resolve(selection)

    def _remove_missing_columns(self) -> None:
        """Remove columns that are not present in the input file.

        Columns are removed from `self._features` and `self._truth`.
        """
        # Check if table is completely empty
        if len(self) == 0:
            self.warning("Dataset is empty.")
            return

        # Find missing features
        missing_features_set = set(self._features)
        for pulsemap in self._pulsemaps:
            missing = self._check_missing_columns(self._features, pulsemap)
            missing_features_set = missing_features_set.intersection(missing)

        missing_features = list(missing_features_set)

        # Find missing truth variables
        missing_truth_variables = self._check_missing_columns(
            self._truth, self._truth_table
        )

        # Remove missing features
        if missing_features:
            self.warning(
                "Removing the following (missing) features: "
                + ", ".join(missing_features)
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

        # Remove missing truth variables
        if missing_truth_variables:
            self.warning(
                (
                    "Removing the following (missing) truth variables: "
                    + ", ".join(missing_truth_variables)
                )
            )
            for missing_truth_variable in missing_truth_variables:
                self._truth.remove(missing_truth_variable)

    def _check_missing_columns(
        self,
        columns: List[str],
        table: str,
    ) -> List[str]:
        """Return a list missing columns in `table`."""
        for column in columns:
            try:
                self.query_table(table, [column], 0)
            except ColumnMissingException:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)
            except IndexError:
                self.warning(f"Dataset contains no entries for {column}")

        return self._missing_variables.get(table, [])

    def _query(
        self, sequential_index: int
    ) -> Tuple[
        List[Tuple[float, ...]],
        Tuple[Any, ...],
        Optional[List[Tuple[Any, ...]]],
        Optional[float],
    ]:
        """Query file for event features and truth information.

        The returned lists have lengths corresponding to the number of pulses
        in the event. Their constituent tuples have lengths corresponding to
        the number of features/attributes in each output

        Args:
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`.

        Returns:
            Tuple containing pulse-level event features; event-level truth
                information; pulse-level truth information; and event-level
                loss weights, respectively.
        """
        features = []
        for pulsemap in self._pulsemaps:
            features_pulsemap, _ = self.query_table(
                pulsemap, self._features, sequential_index, self._selection
            )
            features.extend(features_pulsemap)

        truth_query = self.query_table(
            self._truth_table, self._truth, sequential_index
        )
        truth: Tuple[Any, ...] = truth_query[0][0]
        scramble_class = truth_query[1]
        
        # add scramble_class to truth
        truth = truth + (scramble_class,)
        

        if self._node_truth:
            assert self._node_truth_table is not None
            node_truth, _ = self.query_table(
                self._node_truth_table,
                self._node_truth,
                sequential_index,
                self._selection,
            )
        else:
            node_truth = None

        loss_weight: Optional[float] = None  # Default
        if self._loss_weight_column is not None:
            assert self._loss_weight_table is not None
            loss_weight_list = self.query_table(
                self._loss_weight_table,
                self._loss_weight_column,
                sequential_index,
            )
            if len(loss_weight_list):
                loss_weight = loss_weight_list[0][0]
            else:
                loss_weight = -1.0

        return features, truth, node_truth, loss_weight

    def _create_graph(
        self,
        features: List[Tuple[float, ...]],
        truth: Tuple[Any, ...],
        node_truth: Optional[List[Tuple[Any, ...]]] = None,
        loss_weight: Optional[float] = None,
    ) -> Data:
        """Create Pytorch Data (i.e. graph) object.

        Args:
            features: List of tuples, containing event features.
            truth: List of tuples, containing truth information.
            node_truth: List of tuples, containing node-level truth.
            loss_weight: A weight associated with the event for weighing the
                loss.

        Returns:
            Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {
            key: truth[index] for index, key in enumerate(self._truth)
        }
        # include scrambled_class in truth_dict
        truth_dict['scrambled_class'] = truth[-1]

        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            assert self._node_truth is not None
            node_truth_dict = {
                key: node_truth_array[:, index]
                for index, key in enumerate(self._node_truth)
            }

        # Create list of truth dicts with labels
        truth_dicts = [labels_dict, truth_dict]
        if node_truth is not None:
            truth_dicts.append(node_truth_dict)

        # Catch cases with no reconstructed pulses
        if len(features):
            node_features = np.asarray(features)[
                :, 1:
            ]  # first entry is index column
        else:
            node_features = np.array([]).reshape((0, len(self._features) - 1))

        # Construct graph data object
        assert self._graph_definition is not None
        graph = self._graph_definition(
            input_features=node_features,
            input_feature_names=self._features[
                1:
            ],  # first entry is index column
            truth_dicts=truth_dicts,
            custom_label_functions=self._label_fns,
            loss_weight_column=self._loss_weight_column,
            loss_weight=loss_weight,
            loss_weight_default_value=self._loss_weight_default_value,
            data_path=self._path,
        )
        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        if "pid" in truth_dict.keys():
            abs_pid = abs(truth_dict["pid"])
            sim_type = truth_dict["sim_type"]

            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": int(abs_pid == 13),
                "muon_stopped": int(truth_dict.get("stopped_muon") == 1),
                "noise": int((abs_pid == 1) & (sim_type != "data")),
                "neutrino": int(
                    (abs_pid != 13) & (abs_pid != 1)
                ),  # @TODO: `abs_pid in [12,14,16]`?
                "v_e": int(abs_pid == 12),
                "v_u": int(abs_pid == 14),
                "v_t": int(abs_pid == 16),
                "track": int(
                    (abs_pid == 14) & (truth_dict["interaction_type"] == 1)
                ),
                "dbang": self._get_dbang_label(truth_dict),
                "corsika": int(abs_pid > 20),
            }
        else:
            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": -1,
                "muon_stopped": -1,
                "noise": -1,
                "neutrino": -1,
                "v_e": -1,
                "v_u": -1,
                "v_t": -1,
                "track": -1,
                "dbang": -1,
                "corsika": -1,
            }
        return labels_dict

    def _get_dbang_label(self, truth_dict: Dict[str, Any]) -> int:
        """Get label for double-bang classification."""
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1

class freedom_SQLiteDataset(freedom_Dataset):
    """Pytorch dataset for reading data from SQLite databases."""

    # Implementing abstract method(s)
    def _init(self) -> None:
        # Check(s)
        self._database_list: Optional[List[str]]
        if isinstance(self._path, list):
            self._database_list = self._path
            self._all_connections_established = False
            self._all_connections: List[sqlite3.Connection] = []
        else:
            self._database_list = None
            assert isinstance(self._path, str)
            assert self._path.endswith(
                ".db"
            ), f"Format of input file `{self._path}` is not supported."

        if self._database_list is not None:
            self._current_database: Optional[int] = None

        # Set custom member variable(s)
        self._features_string = ", ".join(self._features)
        self._truth_string = ", ".join(self._truth)
        if self._node_truth:
            self._node_truth_string = ", ".join(self._node_truth)

        self._conn: Optional[sqlite3.Connection] = None

    def _post_init(self) -> None:
        self._close_connection()

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        """Query table at a specific index, optionally with some selection."""
        # Check(s)
        if isinstance(columns, list):
            columns = ", ".join(columns)

        if not selection:  # I.e., `None` or `""`
            selection = "1=1"  # Identically true, to select all

        index = self._get_event_index(sequential_index)
        scramble_class = self._get_event_scramble_class(sequential_index)

        # Query table
        assert index is not None
        assert scramble_class is not None
        self._establish_connection(index)
        try:
            assert self._conn
            if sequential_index is None:
                combined_selections = selection
            else:
                combined_selections = (
                    f"{self._index_column} = {index} and {selection}"
                )

            result = self._conn.execute(
                f"SELECT {columns} FROM {table} WHERE "
                f"{combined_selections}"
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" in str(e):
                raise ColumnMissingException(str(e))
            else:
                raise e
        return result, scramble_class

    def _get_all_indices(self) -> List[tuple]:

        print('getting all indices')
        self._establish_connection(0)
        indices = pd.read_sql_query(
            f"SELECT {self._index_column} FROM {self._truth_table}", self._conn
        )
        self._close_connection()

        # Filter based on pulse count
        conn = sqlite3.connect(self._path)
        query = f"SELECT event_no FROM {pulsemap} WHERE event_no IN ({','.join(map(str, indices))}) GROUP BY event_no HAVING COUNT(*) BETWEEN ? AND ?"

        min_count = 150
        max_count = 300
        indices = [event_no for event_no, in conn.execute(query, (min_count, max_count)).fetchall()]
        print('done')
        return [(num, 0) for num in indices] + [(num, 1) for num in indices]

    def _get_event_index(
        self, sequential_index: Optional[int]
    ) -> Optional[int]:
        index: int = 0
        if sequential_index is not None:
            index_ = self._indices[sequential_index][0]
            if self._database_list is None:
                assert isinstance(index_, int)
                index = index_
            else:
                assert isinstance(index_, tuple)
                index = index_[0]
        return index
    
    def _get_event_scramble_class(
        self, sequential_index: Optional[int]
    ) -> Union[int, list]:
        scramble_class: int = 1
        if sequential_index is not None:
            scramble_class_ = self._indices[sequential_index]
            if self._database_list is None:
                assert isinstance(scramble_class_, tuple)
                scramble_class = scramble_class_[1]
            else:
                assert isinstance(scramble_class_, list)
                scramble_class = scramble_class_[0][1]
        else:
            scramble_class_ = self._indices
            if self._database_list is None:
                assert isinstance(scramble_class_, list)
                scramble_class = np.array(scramble_class_)[:,1].tolist()
            else:
                assert isinstance(scramble_class_, list)
                scramble_class = np.array(scramble_class_[0])[:,1].tolist()
        return scramble_class

    # Custom, internal method(s)
    # @TODO: Is it necessary to return anything here?
    def _establish_connection(self, i: int) -> "SQLiteDataset":
        """Make sure that a sqlite3 connection is open."""
        if self._database_list is None:
            assert isinstance(self._path, str)
            if self._conn is None:
                self._conn = sqlite3.connect(self._path)
        else:
            indices = self._indices[i]
            assert isinstance(indices, list)
            if self._conn is None:
                if self._all_connections_established is False:
                    self._all_connections = []
                    for database in self._database_list:
                        con = sqlite3.connect(database)
                        self._all_connections.append(con)
                    self._all_connections_established = True
                self._conn = self._all_connections[indices[1]]
            if indices[1] != self._current_database:
                self._conn = self._all_connections[indices[1]]
                self._current_database = indices[1]
        return self

    # @TODO: Is it necessary to return anything here?
    def _close_connection(self) -> "SQLiteDataset":
        """Make sure that no sqlite3 connection is open.

        This is necessary to calls this before passing to
        `torch.DataLoader` such that the dataset replica on each worker
        is required to create its own connection (thereby avoiding
        `sqlite3.DatabaseError: database disk image is malformed` errors
        due to inability to use sqlite3 connection accross processes.
        """
        if self._conn is not None:
            self._conn.close()
            del self._conn
            self._conn = None
        if self._database_list is not None:
            if self._all_connections_established:
                for con in self._all_connections:
                    con.close()
                del self._all_connections
                self._all_connections_established = False
                self._conn = None
        return self

def make_freedom_dataloader(
    db: str,
    pulsemaps: Union[str, List[str]],
    graph_definition: GraphDefinition,
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    shuffle: bool,
    selection: Optional[List[int]] = None,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: List[str] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: List[int] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
    no_of_events: Optional[int] = None,
    seed = 1,
) -> DataLoader:
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]


    dataset = freedom_SQLiteDataset(
        path=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        selection=selection,
        node_truth=node_truth,
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_table=loss_weight_table,
        loss_weight_column=loss_weight_column,
        index_column=index_column,
        graph_definition=graph_definition,
    )
    if no_of_events is not None:
        selection = dataset._get_all_indices()
        random.seed(seed)
        selection = random.sample(selection, no_of_events)
        
        dataset = freedom_SQLiteDataset(
            path=db,
            pulsemaps=pulsemaps,
            features=features,
            truth=truth,
            selection=selection,
            node_truth=node_truth,
            truth_table=truth_table,
            node_truth_table=node_truth_table,
            string_selection=string_selection,
            loss_weight_table=loss_weight_table,
            loss_weight_column=loss_weight_column,
            index_column=index_column,
            graph_definition=graph_definition,
        )
        
    
    # adds custom labels to dataset
    if isinstance(labels, dict):
        for label in labels.keys():
            dataset.add_label(key=label, fn=labels[label])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )

    return dataloader

class ScrambledDirection(Label):
    """Class for producing particle direction/pointing label and randomly it based on scramble_flag."""

    def __init__(
        self,
        key: str = "scrambled_direction",
        scramble_flag: str = 'scrambled_class',
        azimuth_key: str = "azimuth",
        zenith_key: str = "zenith",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            scramble_flag: The name of the pre-existing key in 'graph' which
                determines whether the constructed 'Direction' will be shuffled.
            azimuth_key: The name of the pre-existing key in `graph` that will
                be used to access the azimuth angle, used when calculating
                the direction.
            zenith_key: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        print('creating scrambled direction label')
        self._scramble_flag = scramble_flag
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # intialize random direction for first call with raised flag
        zenith = torch.rand((1,))*torch.pi
        azimuth = torch.rand((1,))*2*torch.pi
        x = torch.cos(azimuth) * torch.sin(zenith).reshape(-1, 1)
        y = torch.sin(azimuth) * torch.sin(zenith).reshape(-1, 1)
        z = torch.cos(zenith).reshape(-1, 1)
        self.direction = torch.cat((x, y, z), dim=1)

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""

        # check that the flag is there
        assert  self._scramble_flag in graph.keys()
        assert graph[self._scramble_flag] is not None

        x = torch.cos(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        z = torch.cos(graph[self._zenith_key]).reshape(-1, 1)

        if graph[self._scramble_flag] == 1:
            val = torch.cat((x, y, z), dim=1)
        else:
            val = self.direction
            self.direction = torch.cat((x, y, z), dim=1)
        return val


path = "/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_2.db"
pulsemap = 'InIcePulses'
target = 'scrambled_class'
truth_table = 'truth'
gpus = [1]
max_epochs = 30
early_stopping_patience = 5
batch_size = 500
num_workers = 30
wandb =  False

features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86

graph_definition = KNNGraph(detector=IceCube86())

labels = {'scrambled_direction': ScrambledDirection(
        zenith_key='zenith',azimuth_key='azimuth'
        )
    }

model = Model.load('/scratch/users/mbranden/graphnet/data/examples/output/train_model_without_configs/dev_northern_tracks_full_part_1/dynedge_scrambled_class_example/model.pth')

skymap_dataloader = make_freedom_dataloader(db=path,
    graph_definition=graph_definition,
    pulsemaps=pulsemap,
    features=features,
    truth=truth,
    batch_size=batch_size,
    num_workers=num_workers,
    truth_table=truth_table,
    labels= labels,
    selection= None, #either None, str, or List[(event_no,scramble_class)]
    no_of_events = 1,
    shuffle = True,
    seed = 5
)

def new_predict_skymap(model, data: Union[Data, List[Data]]) -> List[Union[torch.Tensor, Data]]:
    """Forward pass, chaining model components."""
    model.inference()
    model.train(mode=False)

    if isinstance(data, Data):
        data = [data]

    truth_azimuth = []
    truth_zenith = []
    event_nos = []

    zenith = np.linspace(0,np.pi,500)
    azimuth = np.linspace(0,2*np.pi,500)
    ze,az = np.meshgrid(zenith, azimuth)
    zenith = torch.tensor(ze.flatten())
    azimuth = torch.tensor(az.flatten())

    x = torch.cos(azimuth) * torch.sin(zenith)
    y = torch.sin(azimuth) * torch.sin(zenith)
    z = torch.cos(zenith)
    directions = torch.stack((x, y, z), dim=1)

    npix = directions.shape[0]
    x_list = []
    

    for d in tqdm(data):
        x = model.backbone(d)
        truth_azimuth.extend(d['azimuth'].numpy())
        truth_zenith.extend(d['zenith'].numpy())
        event_nos.extend(d['event_no'].numpy())

        for z in range(x.shape[0]):
            x_list.extend(npix*[x[z]])
    
    
    events_count = int(len(x_list)/npix)
    y_list = [directions for _ in range(events_count)]
    x = torch.stack(x_list)
    y = torch.stack(y_list).reshape(len(x_list),3)

    # Add scrambled target to inputs
    x = torch.cat([x, y], dim=1).float()  # Shape: (num_events * npix, feature_dim + 3)
    
    # Pass both latent vec and scrambled target to discriminator
    x = model._discriminator(x)

    # Pass to task
    task_preds = [task(x) for task in model._tasks]
    events_count = len(data)
    pred_chunk = task_preds[0].chunk(events_count)  # Only takes first task for now
    preds = np.array([event_pred.detach().numpy() for event_pred in pred_chunk])

    return preds, az, ze, truth_azimuth, truth_zenith, event_nos

skymap, azimuth, zenith, truth_azimuth, truth_zenith, event_nos = new_predict_skymap(model,skymap_dataloader)
log_skymap = np.log(skymap[0].reshape(azimuth.shape))
max_log_skymap = np.max(log_skymap)
delta_log_skymap = log_skymap - max_log_skymap

sqliteConnection = sqlite3.connect("/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_2.db")
cursor = sqliteConnection.cursor()
event_numbers_str = ','.join(map(str, event_nos))
query = f"SELECT * FROM spline_mpe_ic WHERE event_no IN ({event_numbers_str});"
cursor.execute(query)
spline = cursor.fetchall()
spline_az = spline[0][0]
spline_ze = spline[0][2]

print('max lr azimuth', azimuth.flatten()[int(np.argmax(skymap[0].flatten()))])
print('max lr zenith', zenith.flatten()[int(np.argmax(skymap[0].flatten()))])
print('max lr = ', max(skymap[0].flatten()))
print('index =', np.argmax(skymap[0].flatten()))
print(truth_azimuth,truth_zenith)


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)  # Remove the projection='mollweide'

# Calculate the index of maximum likelihood
max_likelihood_idx = np.argmax(np.log(skymap[0].flatten()))

# Define the zoom region (20 degrees in radians)
delta = 20 * (np.pi / 180)

# Calculate the azimuth and zenith bounds for the zoomed-in region
azimuth_min = azimuth.flatten()[max_likelihood_idx] - delta
azimuth_max = azimuth.flatten()[max_likelihood_idx] + delta
zenith_min = zenith.flatten()[max_likelihood_idx] - delta
zenith_max = zenith.flatten()[max_likelihood_idx] + delta

# Mask the data within the zoomed-in region
mask = (azimuth >= azimuth_min) & (azimuth <= azimuth_max) & \
       (zenith >= zenith_min) & (zenith <= zenith_max)

masked_log_skymap = np.ma.masked_where(~mask, delta_log_skymap)

# Find the min and max values within the masked region for color scaling
min_val = np.min(masked_log_skymap)
max_val = np.max(masked_log_skymap)

# Plot the masked skymap with the adjusted colormap scale
cax = ax.pcolormesh(azimuth - np.pi, -zenith + np.pi / 2, delta_log_skymap,
                    cmap=cm.viridis, shading='gouraud', vmin=min_val, vmax=max_val)

# Add colorbar
cb = fig.colorbar(cax, orientation='horizontal')
cb.set_label(r'$\Delta$log-likelihood')

# Plot the truth position and max likelihood position
ax.plot(truth_azimuth[0] - np.pi, -truth_zenith[0] + np.pi / 2, 'rx', markersize=10, label='Truth')
ax.plot(spline_az - np.pi, -spline_ze + np.pi / 2, 'gx', markersize=10, label='spline mpe')
ax.plot(azimuth.flatten()[max_likelihood_idx] - np.pi, -zenith.flatten()[max_likelihood_idx] + np.pi / 2, 'bx', markersize=10, label='Max Likelihood')

# Add a contour line
clevel = -2.305
cs = ax.contour(azimuth - np.pi, -zenith + np.pi / 2, delta_log_skymap, levels=[clevel], linewidths=1, linestyles='solid')
# print(cs.levels)
# fmt = {}
# strs = ['90 %']
# for l, s in zip(cs.levels, strs):
#     fmt[l] = s
# print(fmt)
# ax.clabel(cs, inline=True,fmt = fmt, fontsize=10)
cs.collections[0].set_label(f'90% contour')

# Set the axis limits to zoom in
ax.set_xlim(azimuth_min - np.pi, azimuth_max - np.pi)
ax.set_ylim(-zenith_max + np.pi / 2, -zenith_min + np.pi / 2)

# Add grid, labels, and legend
ax.grid(True)
ax.set_xlabel('Azimuth [rad]')
ax.set_ylabel('Zenith [rad]')
ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('./plots/delta_llh_skymap_zoomed_1.pdf')
plt.show()
plt.close()