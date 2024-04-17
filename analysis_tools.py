from typing import Optional, cast

import numpy as np
from colouring_utils import Colour, ColourScheme, make_diverging_colormap
import data
import formulas

# Non standard libraries will need to be installed within
# PyMol itself, so the script can run in that environment.
try:
    from scipy import stats
except ImportError as exc:
    raise ImportError("Please install scipy via 'pip install scipy'.") from exc


def compare_uptakes(
    default: data.Uptake,
    other: data.Uptake,
    params: data.UserParameters,
    SEM_for_hybrid_test: Optional[float] = None,
) -> tuple[float, float, bool]:

    uptake_difference = other.mean - default.mean

    p_value = stats.ttest_ind_from_stats(
        mean1=default.mean,
        std1=default.stdev,
        nobs1=params.n_repeats,
        mean2=other.mean,
        std2=other.stdev,
        nobs2=params.n_repeats,
    ).pvalue

    significant = p_value < (1 - params.confidence_interval)

    if params.statistical_test == data.StatisticalTestType.HYBRID:
        assert (
            SEM_for_hybrid_test is not None
        ), "Must provide the standard error of the mean to perform a hybrid test."
        degrees_of_freedom = (params.n_repeats * 2) - 2
        t_critical = stats.t.ppf(params.confidence_interval, degrees_of_freedom)
        globally_significant = abs(uptake_difference) > (
            SEM_for_hybrid_test * t_critical
        )
        significant &= globally_significant

    return uptake_difference, p_value, significant


def compare_states(
    default: data.StateData,
    other: data.StateData,
    params: data.UserParameters,
    SEM_for_hybrid_test: float,
) -> dict[str, data.Comparison]:

    if not default.is_same_fragment(other):
        raise ValueError(
            f"Cannot compare state data from different sequences ({default} != {other})."
        )

    results_by_exposure: dict[str, data.Comparison] = {}

    # This is including the cumulative data as an exposure "timepoint"
    for (default_exposure, default_uptake), (other_exposure, other_uptake) in zip(
        default.exposures.items(), other.exposures.items()
    ):
        if default_exposure != other_exposure:
            raise ValueError(
                f"States for fragment {default.sequence} do not have the same order exposures."
            )

        uptake_difference, p_value, is_significant = compare_uptakes(
            default=default_uptake,
            other=other_uptake,
            params=params,
            SEM_for_hybrid_test=SEM_for_hybrid_test,
        )

        comparison = data.Comparison.from_reference(
            reference=default,
            uptake_difference=uptake_difference,
            p_value=p_value,
            is_significant=is_significant,
            exposure=default_exposure,
        )

        results_by_exposure[default_exposure] = comparison

    return results_by_exposure


def split_comparisons_by_residue(
    comparisons: list[data.Comparison], params: data.ExperimentalParameters
) -> list[data.SingleResidueComparison]:
    assert (
        len(set([c.exposure for c in comparisons])) == 1
    ), "Comparisons must be from the same exposure time."
    exposure = comparisons[0].exposure

    all_residues = np.arange(params.max_residue + 1)

    single_residue_comparisons: list[data.SingleResidueComparison] = []
    for residue in all_residues:

        residue_present = [
            comparison
            for comparison in comparisons
            if comparison.residue_present(residue)
        ]
        # Residue is not covered by any sequences, therefore missed in the digestion process.
        if len(residue_present) == 0:
            single_residue_comparisons.append(
                data.SingleResidueComparison.as_empty(
                    residue=residue, exposure=exposure
                )
            )
            continue

        amino_acid: str = residue_present[0].sequence[
            residue - residue_present[0].start_residue
        ]

        is_significant = [
            comparison for comparison in residue_present if comparison.is_significant
        ]

        single_residue_comparisons.append(
            data.SingleResidueComparison(
                amino_acid=amino_acid,
                residue=residue,
                uptake_difference=(
                    np.nan
                    if len(is_significant) == 0
                    else cast(
                        float, np.mean([i.uptake_difference for i in is_significant])
                    )  # TODO: I'm not particularly sure if this makes sense mathematically...
                ),
                uptake_stdev=(
                    np.nan
                    if len(is_significant) == 0
                    else cast(
                        float, np.std([i.uptake_difference for i in is_significant])
                    )  # TODO: I'm not particularly sure if this makes sense mathematically...
                ),
                exposure=exposure,
                residue_type=(
                    data.ResidueType.ALL_INSIGNIFICANT
                    if len(is_significant) == 0
                    else data.ResidueType.AVERAGED
                ),
            )
        )

    return single_residue_comparisons


def get_strongest_magnitude_of_type(
    comparisons: list[data.Comparison], data_type: data.DataForVisualisation
) -> float:
    values = [abs(comparison.request(data_type)) for comparison in comparisons]
    return max(values)


def find_normalisation_value(
    comparisons: dict[str, list[data.Comparison]],
    data_type: data.DataForVisualisation,
    normalisation_mode: data.NormalisationMode,
) -> float:
    if normalisation_mode == data.NormalisationMode.GLOBAL:
        rel_data = [
            comparison
            for collection in comparisons.values()
            for comparison in collection
        ]
    elif normalisation_mode == data.NormalisationMode.ACROSS_EXPOSURES:
        rel_data = [
            comparison
            for (exposure, collection) in comparisons.items()
            for comparison in collection
            if exposure != data.CUMULATIVE_EXPOSURE_KEY
        ]
    else:
        raise ValueError(f"Cannot interpret normalisation mode {normalisation_mode}")

    return get_strongest_magnitude_of_type(rel_data, data_type)


def run_SAUSC_from_path(
    filepath: str,
    n_repeats: int,
    confidence_interval: float,
    hybrid_test: bool,
    protection_colourmap: str,
    deprotection_colourmap: str,
    insignificant_colour: Colour,
    no_coverage_colour: Colour,
    global_normalisation: bool,
) -> data.FullSAUSCAnalysis:

    user_params = data.UserParameters(
        n_repeats=n_repeats,
        confidence_interval=confidence_interval,
        global_normalisation=global_normalisation,
        statistical_test=(
            data.StatisticalTestType.HYBRID
            if hybrid_test
            else data.StatisticalTestType.T_TEST
        ),
    )
    colour_scheme = ColourScheme(
        uptake_colourmap=make_diverging_colormap(
            protection_colormap=protection_colourmap,
            deprotection_colormap=deprotection_colourmap,
        ),
        insignificant=insignificant_colour,
        no_coverage=no_coverage_colour,
    )
    loaded_results, experimental_params = data.load_state_data(filepath)

    global_sem = formulas.pooled_standard_error_mean(
        stdevs=np.array(
            [
                uptake.stdev
                for data in loaded_results
                for uptake in data.exposures.values()
            ]
        ),
        n_repeats=user_params.n_repeats,
    )
    
    # Hopefully a user will call the default state "default"...
    if "default" in experimental_params.states[0].lower():
        default_is_first = True
    elif "default" in experimental_params.states[1].lower():
        default_is_first = False
    else:
        raise ValueError("Cannot determine which state is the default!")

    # The results should be structured interlaced
    default_states, other_states = np.reshape(np.array(loaded_results), (-1, 2)).T

    if not default_is_first:
        default_states, other_states = other_states, default_states

    _comparisons: list[dict[str, data.Comparison]] = [
        compare_states(default, other, user_params, global_sem)
        for default, other in zip(default_states, other_states)
    ]

    comparisons_by_exposure: dict[str, list[data.Comparison]] = {
        exposure: [comparison[exposure] for comparison in _comparisons]
        for exposure in _comparisons[0].keys()
    }

    single_residue_comparisons: dict[str, list[data.SingleResidueComparison]] = {
        exposure: split_comparisons_by_residue(comparisons, experimental_params)
        for exposure, comparisons in comparisons_by_exposure.items()
    }
    protein_sequence = "".join(
        [
            residue.amino_acid
            for residue in list(single_residue_comparisons.values())[0]
        ]
    )

    full_analysis = data.FullSAUSCAnalysis(
        user_params=user_params,
        experimental_params=experimental_params,
        sequence_comparisons=comparisons_by_exposure,
        residue_comparisons=single_residue_comparisons,
        colouring=colour_scheme,
        full_sequence=protein_sequence,
        global_standard_error_mean=global_sem
    )
    return full_analysis