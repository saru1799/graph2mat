"""Implements classes to store an example of the dataset in memory.

A "configuration" is an object that contains all the information about a
given example in the dataset. It contains all the features needed to
describe the example (e.g. coordinates, lattice vectors...), and optionally
the matrix that corresponds to this example.

In a typical case, your configurations will contain the matrix as a label
for training, validating or testing. When doing inference, the configurations
will not have an associated matrix, since the matrix is what you are trying
to calculate.
"""
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
import sisl
from scipy.sparse import csr_array, issparse

from .basis import NoBasisAtom, PointBasis
from .formats import Formats, conversions
from .matrices import BasisMatrix, OrbitalMatrix, get_matrix_cls
from .sparse import csr_to_block_dict
from .table import BasisTableWithEdges

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)
PhysicsMatrixType = Literal[
    "density_matrix", "hamiltonian", "energy_density_matrix", "dynamical_matrix"
]

DEFAULT_CONFIG_TYPE = "Default"


@dataclass
class BasisConfiguration:
    """Container class to store all the information of an example.


    Stores a distribution of points in space, with associated basis functions.
    Optionally, it can also store an associated matrix.

    In a typical case, your configurations will contain the matrix as a label
    for training, validating or testing. When doing inference, the configurations
    will not have an associated matrix, since the matrix is what you are trying
    to calculate.

    This is a `dataclasses.dataclass`. It is purely a container for the information
    of one example in your dataset.

    Parameters
    -----------
    point_types:
        Shape (n_points,).
        The type of each point. Each type can be either a string or an integer,
        and it should be the type key of a `PointBasis` object in the `basis` list.
    positions:
        Shape (n_points, 3).
        The positions of each point in cartesian coordinates.
    basis:
        List of `PointBasis` objects for types that are (possibly) present in the system.
    cell:
        Shape (3, 3).
        The cell vectors that delimit the system, in cartesian coordinates.
    pbc:
        Shape (3,).
        Whether the system is periodic in each cell direction.
    matrix:
        The matrix associated to the configuration.

        It can be a numpy or scipy sparse matrix, which will be converted to a BasisMatrix
        object.
    weight:
        The weight of the configuration in the loss.
    config_type:
        A string that indicates the type of configuration.
    metadata:
        A dictionary with additional metadata related to the configuration.
    """

    #: Shape (n_points,).
    #: The type of each point. Each type can be either a string or an integer,
    #: and it should be the type key of a `PointBasis` object in the `basis` list.
    point_types: np.ndarray

    #: Shape (n_points, 3).
    #: The positions of each point in cartesian coordinates.
    positions: Positions

    #: List of `PointBasis` objects for types that are (possibly) present in the system.
    basis: Sequence[PointBasis]

    #: Shape (3, 3).
    #: The cell vectors that delimit the system, in cartesian coordinates.
    cell: Optional[Cell] = None

    #: Shape (3,).
    #: Whether the system is periodic in each cell direction.
    pbc: Optional[Pbc] = None

    #: The matrix associated to the configuration.
    matrix: Optional[BasisMatrix] = None

    #: The weight of the configuration in the loss.
    weight: float = 1.0

    #: A string that indicates the type of configuration.
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE

    #: A dictionary with additional metadata related to the configuration.
    metadata: Optional[Dict[str, Any]] = None

    #: The graph2mat format that identifies this class.
    _cls_format: str = Formats.BASISCONFIGURATION

    def __post_init__(self):
        # SN: Warning! This only works properly with square matrices.
        if isinstance(self.basis, BasisTableWithEdges):
            # If the basis is a table, we need to convert it to a list of PointBasis objects
            object.__setattr__(self, "basis", self.basis.row_basis)
        elif isinstance(self.basis, dict):
            # If the basis is a dict, we need to convert it to a list of PointBasis objects
            # Again, only take rows
            object.__setattr__(self, "basis", self.basis["row"])

        if self.matrix is not None and not isinstance(self.matrix, BasisMatrix):
            matrix = self.matrix

            if isinstance(self.matrix, np.ndarray):
                matrix = csr_array(matrix)

            if issparse(matrix):
                matrix = sisl.SparseCSR(matrix)

                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError(
                        f"{self.__class__.__name__} can only sanitize the provided matrix if it is square. Otherwise use `BasisConfiguration.from_matrix`"
                    )

                geometry = self.to_sisl_geometry()
                matrix = csr_to_block_dict(
                    matrix, geometry.atoms, nsc=(1, 1, 1), matrix_cls=BasisMatrix
                )

                object.__setattr__(self, "matrix", matrix)

    def to_sisl_geometry(self) -> sisl.Geometry:
        """Converts the configuration to a sisl Geometry."""
        return conversions.get_converter(
            Formats.BASISCONFIGURATION, Formats.SISL_GEOMETRY
        )(self)

    @classmethod
    def new(
        cls,
        obj: Union[sisl.Geometry, sisl.SparseOrbital, str, Path],
        labels: bool = True,
        **kwargs,
    ) -> "BasisConfiguration":
        """Creates a new configuration.

        This is just a dispatcher that will call the appropriate method to create
        the object depending on the type of the input.

        Parameters
        -----------
        obj:
            The object from which to create the configuration.
        labels:
            Whether to find labels (the matrix) to be assigned to the configuration.
        **kwargs:
            Additional arguments to be passed to the constructor of the configuration.

        See Also
        ---------
        from_geometry, from_matrix, from_run
            The methods that are called by this dispatcher to create the new configuration,
            depending on the type of `obj`.
        """
        if isinstance(obj, sisl.Geometry):
            if labels:
                raise ValueError(
                    "Cannot infer output labels only from a geometry. Please provide either a matrix or a path to a run file."
                )
            return cls.from_geometry(obj, **kwargs)
        elif isinstance(obj, sisl.SparseOrbital):
            return cls.from_matrix(obj, labels=labels, **kwargs)
        elif isinstance(obj, (str, Path, zipfile.Path)):
            if not labels:
                kwargs["out_matrix"] = None
            return cls.from_run(obj, **kwargs)
        else:
            raise TypeError(
                f"Cannot create {cls.__name__} from {obj.__class__.__name__}."
            )

    @classmethod
    def from_geometry(cls, geometry: sisl.Geometry, **kwargs) -> "BasisConfiguration":
        """Initializes a configuration from a sisl geometry.

        Note that the created object will not have an associated matrix, unless it is passed
        explicitly as a keyword argument.

        Parameters
        -----------
        geometry: sisl.Geometry
            The geometry to associate to the configuration.
        **kwargs:
            Additional arguments to be passed to the configuration constructor.
        """
        converter = conversions.get_converter(Formats.SISL_GEOMETRY, cls._cls_format)
        return converter(geometry, cls=cls, **kwargs)

    @classmethod
    def from_matrix(
        cls,
        matrix: sisl.SparseOrbital,
        geometry: Union[sisl.Geometry, None] = None,
        labels: bool = True,
        **kwargs,
    ) -> "BasisConfiguration":
        """Initializes a configuration from a sisl matrix.

        Parameters
        -----------
        matrix: sisl.SparseOrbital
            The matrix to associate to the configuration. This matrix should have an associated
            geometry, which will be used.
        geometry: sisl.Geometry, optional
            The geometry to associate to the configuration. If None, the geometry of the matrix
            will be used.
        labels: bool
            Whether to process the labels from the matrix. If False, the only thing to read
            will be the atomic structure, which is likely the input of your model.
        **kwargs:
            Additional arguments to be passed to the configuration constructor.
        """
        converter = conversions.get_converter(Formats.SISL, cls._cls_format)
        return converter(matrix, geometry=geometry, labels=labels, cls=cls, **kwargs)

    @classmethod
    def from_run(
        cls,
        runfilepath: Union[str, Path],
        geometry_path: Optional[Union[str, Path]] = None,
        out_matrix: Optional[PhysicsMatrixType] = None,
        basis: Optional[sisl.Atoms] = None,
    ) -> "BasisConfiguration":
        """Initializes configuration from the main input file of a run.

        Parameters
        -----------
        runfilepath:
            The path of the main input file. E.g. in SIESTA this is the path to the ".fdf" file
        geometry_path:
            The path to the geometry file. If None, the geometry will be read from the run file.
        out_matrix:
            The matrix to be read from the output of the run. The configuration object will
            contain the matrix.
            If it is None, then no matrices are read from the output. This is the case when trying to
            predict matrices, since you don't have the output yet.
        basis:
            The basis to use for the configuration. If None, the basis of the read geometry
            will be used.
        """
        converter = conversions.get_converter(Formats.SISL_SILE, cls._cls_format)
        return converter(
            runfilepath,
            geometry_path=geometry_path,
            out_matrix=out_matrix,
            basis=basis,
            cls=cls,
        )


@dataclass
class OrbitalConfiguration(BasisConfiguration):
    """Stores a distribution of atoms in space, with associated orbitals.

    Optionally, it can also store an associated matrix.

    In a typical case, your configurations will contain the matrix as a label
    for training, validating or testing. When doing inference, the configurations
    will not have an associated matrix, since the matrix is what you are trying
    to calculate.

    This is a version of `BasisConfiguration` for atomic systems,
    where points are atoms.

    Parameters
    -----------
    point_types:
        Shape (n_points,).
        The type of each point. Each type can be either a string or an integer,
        and it should be the type key of a `PointBasis` object in the `basis` list.
    positions:
        Shape (n_points, 3).
        The positions of each point in cartesian coordinates.
    basis:
        Atoms that are (possibly) present in the system.
    cell:
        Shape (3, 3).
        The cell vectors that delimit the system, in cartesian coordinates.
    pbc:
        Shape (3,).
        Whether the system is periodic in each cell direction.
    matrix:
        The matrix associated to the configuration.

        It can be a numpy or scipy sparse matrix, which will be converted to a BasisMatrix
        object.
    weight:
        The weight of the configuration in the loss.
    config_type:
        A string that indicates the type of configuration.
    metadata:
        A dictionary with additional metadata related to the configuration.
    """

    #: Shape (n_points,).
    #: The type of each point. Each type can be either a string or an integer,
    #: and it should be the type key of a `PointBasis` object in the `basis` list.
    point_types: np.ndarray

    #: Shape (n_points, 3).
    #: The positions of each point in cartesian coordinates.
    positions: Positions

    #: Atoms that are (possibly) present in the system.
    basis: sisl.Atoms

    #: Shape (3, 3).
    #: The cell vectors that delimit the system, in cartesian coordinates.
    cell: Optional[Cell] = None

    #: Shape (3,).
    #: Whether the system is periodic in each cell direction.
    pbc: Optional[Pbc] = None

    #: The matrix associated to the configuration.
    matrix: Optional[OrbitalMatrix] = None

    #: The weight of the configuration in the loss.
    weight: float = 1.0

    #: A string that indicates the type of configuration.
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE

    #: A dictionary with additional metadata related to the configuration.
    metadata: Optional[Dict[str, Any]] = None

    #: The graph2mat format that identifies this class.
    _cls_format: str = Formats.ORBITALCONFIGURATION

    @property
    def atom_types(self) -> np.ndarray:
        """Alias for point_types."""
        return self.point_types

    @property
    def atoms(self) -> sisl.Atoms:
        """Alias for basis."""
        return self.basis


# ---------------------------------------
#    Add to the formats register
# ---------------------------------------
# Register the classes as aliases for the formats
Formats.add_alias(Formats.BASISCONFIGURATION, BasisConfiguration)
Formats.add_alias(Formats.ORBITALCONFIGURATION, OrbitalConfiguration)

# Conversions from/to BasisConfiguration classes
converter = conversions.converter


@converter
def _orbitalconfiguration_to_basisconfiguration(
    config: OrbitalConfiguration,
) -> BasisConfiguration:
    """Converts an OrbitalConfiguration to a BasisConfiguration.

    The only thing that this function does is to convert the ``basis`` attribute
    from a list of sisl atoms to a list of ``PointBasis`` objects.
    """
    return BasisConfiguration(
        point_types=config.point_types,
        positions=config.positions,
        basis=[PointBasis.from_sisl_atom(p) for p in config.basis],
        cell=config.cell,
        pbc=config.pbc,
        matrix=config.matrix,
        weight=config.weight,
        config_type=config.config_type,
        metadata=config.metadata,
    )


@converter
def _configuration_to_geometry(config: BasisConfiguration) -> sisl.Geometry:
    # BORRAR
    print(config.basis)
    atoms = {pb.type: pb.to_sisl_atom(Z=i + 1) for i, pb in enumerate(config.basis)}

    return sisl.Geometry(
        xyz=config.positions,
        atoms=[atoms[k] for k in config.point_types],
        lattice=config.cell,
    )


@converter(Formats.SISL_GEOMETRY, Formats.ORBITALCONFIGURATION)
def _geometry_to_configuration(
    geometry: sisl.Geometry,
    cls: type[OrbitalConfiguration] = OrbitalConfiguration,
    **kwargs,
) -> OrbitalConfiguration:
    """Initializes an ``OrbitalConfiguration`` object from a sisl geometry.

    Note that the created object will not have an associated matrix, unless it is passed
    explicitly as a keyword argument.

    Parameters
    ----------
    geometry:
        The geometry to associate to the OrbitalConfiguration.
    cls:
        Class to initialize, should be a subclass of ``OrbitalConfiguration``.
    **kwargs:
        Additional arguments to be passed to the OrbitalConfiguration constructor.
    """

    if "pbc" not in kwargs:
        kwargs["pbc"] = (True, True, True)

    return cls(
        point_types=geometry.atoms.Z,
        basis=geometry.atoms,
        positions=geometry.xyz,
        cell=geometry.cell,
        **kwargs,
    )


@converter(Formats.SISL, Formats.ORBITALCONFIGURATION)
def _sisl_to_orbital_configuration(
    matrix: sisl.SparseOrbital,
    geometry: Union[sisl.Geometry, None] = None,
    labels: bool = True,
    cls: type[OrbitalConfiguration] = OrbitalConfiguration,
    **kwargs,
) -> OrbitalConfiguration:
    """Initializes an OrbitalConfiguration object from a sisl matrix.

    Parameters
    ----------
    matrix
        The matrix to associate to the OrbitalConfiguration. This matrix should have an associated
        geometry, which will be used.
    geometry
        The geometry to associate to the OrbitalConfiguration. If None, the geometry of the matrix
        will be used.
    labels
        Whether to process the labels from the matrix. If False, the only thing to read
        will be the atomic structure, which is likely the input of your model.
    cls
        Class to initialize, should be a subclass of ``OrbitalConfiguration``.
    **kwargs
        Additional arguments to be passed to the OrbitalConfiguration constructor.
    """
    if geometry is None:
        # The matrix will have an associated geometry, so we will use it.
        geometry = matrix.geometry

    if labels:
        # Determine the dataclass that should store the matrix and build the block dict
        # sparse structure.
        matrix_cls = get_matrix_cls(matrix.__class__)
        matrix_block = csr_to_block_dict(
            matrix._csr,
            matrix.atoms,
            nsc=matrix.nsc,
            matrix_cls=matrix_cls,
            geometry_atoms=geometry.atoms,
            fill_value=np.nan
            if isinstance(matrix, (sisl.DensityMatrix, sisl.EnergyDensityMatrix))
            else 0.0,
        )

        kwargs["matrix"] = matrix_block

    from_geometry = conversions.get_converter(
        Formats.SISL_GEOMETRY, Formats.ORBITALCONFIGURATION
    )

    return from_geometry(geometry=geometry, cls=cls, **kwargs)


@converter(Formats.SISL_SILE, Formats.ORBITALCONFIGURATION)
def _sisl_run_to_orbitalconfiguration(
    runfilepath: Union[str, Path],
    geometry_path: Optional[Union[str, Path]] = None,
    out_matrix: Optional[PhysicsMatrixType] = None,
    cls: type[OrbitalConfiguration] = OrbitalConfiguration,
    basis: Optional[sisl.Atoms] = None,
) -> OrbitalConfiguration:
    """Initializes an OrbitalConfiguration object from the main input file of a run.

    Parameters
    ----------
    runfilepath:
        The path of the main input file. E.g. in SIESTA this is the path to the ".fdf" file
    geometry_path:
        The path to the geometry file. If None, the geometry will be read from the run file.
    out_matrix:
        The matrix to be read from the output of the run. The configuration object will
        contain the matrix.
        If it is None, then no matrices are read from the output. This is the case when trying to
        predict matrices, since you don't have the output yet.
    cls:
        Class to initialize, should be a subclass of ``OrbitalConfiguration``.
    basis:
        The basis to use for the configuration. If None, the basis of the read geometry
        will be used.
    """
    # Initialize the file object for the main input file
    main_input = sisl.get_sile(runfilepath)
    # Build some metadata so that the OrbitalConfiguration object can be traced back to the run.
    metadata = {"path": str(runfilepath)}

    def _change_geometry_basis(geometry, basis):
        # new_atoms = geometry.atoms.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for atom in [*geometry.atoms.atom]:
                for basis_at in basis:
                    if atom.tag == basis_at.tag:
                        geometry.atoms[atom.tag] = basis_at
                        break
                else:
                    raise ValueError(
                        f"Atom '{atom.tag}' not found in the provided basis"
                    )

    def _read_geometry(main_input, basis):
        # Read the geometry from the main input file
        try:
            geometry = main_input.read_geometry(output=True)
        except TypeError:
            geometry = main_input.read_geometry()

        if basis is not None:
            _change_geometry_basis(geometry, basis)

        return geometry

    def _copy_basis(
        original: sisl.Geometry, geometry: sisl.Geometry, notfound_ok=False
    ) -> sisl.Geometry:
        import warnings

        new_geometry = geometry.copy()

        for atom in geometry.atoms.atom:
            for basis_atom in original.atoms:
                if basis_atom.tag == atom.tag:
                    break
            else:
                if not notfound_ok:
                    raise ValueError(f"Couldn't find atom {atom} in the basis")
                basis_atom = NoBasisAtom(atom.Z, tag=atom.tag)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_geometry.atoms.replace_atom(atom, basis_atom)

        return new_geometry

    if isinstance(main_input, sisl.io.fdfSileSiesta):
        type_of_run = main_input.get("MD.TypeOfRun")
        if type_of_run == "qmmm":
            pipe_file = main_input.get("QMMM.Driver.QMRegionFile")
            # geometry_path = main_input.file.parent / (pipe_file.split(".")[0] + ".last.pdb")
            geometry_path = main_input.file.parent / (pipe_file.split(".")[0] + ".XV")

    if out_matrix is not None:
        # Get the method to read the desired matrix and read it
        read = getattr(main_input, f"read_{out_matrix}")
        if basis is not None:
            geometry = _read_geometry(main_input, basis)
            matrix = read(geometry=geometry)
        else:
            matrix = read()

        kwargs = {}
        if geometry_path is not None:
            # If we have a geometry path, we will read the geometry from there.
            # from ase.io import read

            kwargs["geometry"] = sisl.Geometry.read(geometry_path)
            kwargs["geometry"] = _copy_basis(
                matrix.geometry, kwargs["geometry"], notfound_ok=True
            )

        metadata["geometry"] = kwargs.get("geometry", matrix.geometry)

        # Now build the OrbitalConfiguration object using this matrix.
        from_matrix = conversions.get_converter(
            Formats.SISL, Formats.ORBITALCONFIGURATION
        )
        return from_matrix(matrix=matrix, metadata=metadata, cls=cls, **kwargs)
    else:
        # We have no matrix to read, we will just read the geometry.
        geometry = _read_geometry(main_input, basis)

        if geometry_path is not None:
            # If we have a geometry path, we will read the geometry from there.
            from ase.io import read

            new_geometry = sisl.Geometry.new(read(geometry_path))
            geometry = _copy_basis(geometry, new_geometry, notfound_ok=True)

        metadata["geometry"] = geometry

        # And build the OrbitalConfiguration object using this geometry.
        from_geometry = conversions.get_converter(
            Formats.SISL_GEOMETRY, Formats.ORBITALCONFIGURATION
        )
        return from_geometry(geometry=geometry, metadata=metadata, cls=cls)
