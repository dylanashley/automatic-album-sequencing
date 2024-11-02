#!streamlit run
# -*- coding: ascii -*-

from album_extractor import LSTMAudioFeatureEncoder, Mean
from album_transformer import (
    AutoregressiveTransformerPredictor,
    get_features,
    get_orderings,
    MLPFeatureEncoder,
)
import mimetypes
import numpy as np
import os
import plotly.graph_objects as go
import streamlit as st
import tempfile
from tools import build_template, default_template, fit_values, get_value, plot, scale
import torch
from typing import Callable


def is_audio_file(file):
    mime_type, _ = mimetypes.guess_type(file.name)
    return mime_type and mime_type.startswith("audio/")


def sdistill_direct(
    files: list[str],
    num_orderings: int = 1,
    progress_callback: Callable = None,
    cache: dict[str, float] = dict(),
) -> tuple[list[list[str]], list[float], list[go.Figure], dict[str, float]]:
    feature_encoder = torch.load("direct_feature_encoder.pt")
    ordering_predictor = torch.load("direct_ordering_predictor.pt")
    device = torch.device("cpu")
    feature_encoder.to(device)
    ordering_predictor.to(device)
    feature_encoder.eval()
    ordering_predictor.eval()
    file_features = []
    for i, filename in enumerate(files):
        if progress_callback is not None:
            progress_callback(i, filename)
        if os.path.basename(filename) not in cache:
            cache[os.path.basename(filename)] = get_features(filename)
        file_features.append(cache[os.path.basename(filename)])
    features = torch.stack(file_features, dim=0)
    orderings, logprobs, narrative_features = get_orderings(
        features,
        feature_encoder,
        ordering_predictor,
        num_orderings_to_generate=num_orderings * 3,
        num_choices=num_orderings,
    )

    playlists = [[files[i] for i in ordering] for ordering in orderings]
    values = [
        {files[i]: narrative_features[i] for i in ordering} for ordering in orderings
    ]
    logprobs = list(logprobs)
    plots = [
        plot(playlists[i], values[i], basename=True, plotly=True)
        for i in range(len(playlists))
    ]
    return playlists, logprobs, plots, cache


def sdistill_template(
    files: list[str],
    template: Callable = default_template,
    progress_callback: Callable = None,
    cache: dict[str, float] = dict(),
) -> tuple[list[str], float, go.Figure, dict[str, float]]:
    encoder = torch.load("album_feature_encoder.pt")
    values = dict()
    for i, filename in enumerate(files):
        if progress_callback is not None:
            progress_callback(i, filename)
        if os.path.basename(filename) not in cache:
            cache[os.path.basename(filename)] = get_value(filename, encoder)
        values[filename] = cache[os.path.basename(filename)]
    min_value = min(values.values())
    max_value = max(values.values())
    for k, v in values.items():
        values[k] = scale(v, min_value, max_value, 0, 1)
    playlist, fitting_loss = fit_values(values, template=template)
    return (
        playlist,
        fitting_loss,
        plot(
            playlist,
            values,
            min_value,
            max_value,
            template,
            basename=True,
            plotly=True,
        ),
        cache,
    )


def main():
    header_container = st.container()
    input_container = st.container()
    output_container = st.container()

    # initialize session state
    if "direct_cache" not in st.session_state:
        st.session_state.direct_cache = dict()
    if "direct_logprobs" not in st.session_state:
        st.session_state.direct_logprobs = []
    if "direct_figures" not in st.session_state:
        st.session_state.direct_figures = []
    if "current_direct_figure" not in st.session_state:
        st.session_state.current_direct_figure = 0
    if "templates" not in st.session_state:
        templates = np.load("templates.npz")
        st.session_state.templates = [
            build_template(list(zip(templates["x"], templates["y"][i])))
            for i in range(templates["y"].shape[0])
        ]
    if "template_cache" not in st.session_state:
        st.session_state.template_cache = dict()
    if "template_fitting_losses" not in st.session_state:
        st.session_state.template_fitting_losses = []
    if "template_figures" not in st.session_state:
        st.session_state.template_figures = []
    if "template_ordering" not in st.session_state:
        st.session_state.template_ordering = []
    if "current_template_figure" not in st.session_state:
        st.session_state.current_template_figure = 0

    # draw header
    header_container.markdown(
        """
        # Automatic Album Sequencer

        This app takes a set of music files and automatically orders
        them so that they are arranged to approximate how a
        professional might sequence them. Two approaches are
        demonstrated here. The first is the direct transformer-based
        approach presented in *Automatic Album Sequencing* by Vincent
        Herrmann, Dylan R. Ashley, and Juergen Schmidhuber. The second
        is the contrastive template-based approach presented in *On the
        Distillation of Stories for Transferring Narrative Arcs in
        Collections of Independent Media* by Dylan R. Ashley, Vincent
        Herrmann, Zachary Friggstad, and Juergen Schmidhuber. In both
        cases, after your files are processed, you'll be able to see
        the shape of the resulting playlist by looking at how the
        narrative essence values of the songs change as you go from
        song to song. All of these sequences should be better in
        expectation than a random ordering.

        ---
        """
    )

    # file uploader
    uploaded_files = input_container.file_uploader(
        "Upload Audio Files", accept_multiple_files=True
    )

    if uploaded_files:
        _, center, _ = input_container.columns([1, 1, 1])

        center.text("")  # hack to add a bit of spacing above the button

        # add a process button
        if center.button(
            "Process {} Files".format(len(uploaded_files)), use_container_width=True
        ):
            center.text("")  # hack to add a bit of spacing below the button
            if len(uploaded_files) < 3:
                input_container.error("Please upload at least three audio files.")
            else:
                non_audio_files = [
                    file.name for file in uploaded_files if not is_audio_file(file)
                ]
                if non_audio_files:
                    input_container.error(
                        "The following files are not recognized as audio files: "
                        f"{', '.join(non_audio_files)}"
                    )
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        audio_files = []
                        for file in uploaded_files:
                            file_path = os.path.join(temp_dir, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            audio_files.append(file_path)

                        # clear previous results
                        st.session_state.direct_logprobs = []
                        st.session_state.direct_figures = []
                        st.session_state.current_direct_figure = 0
                        st.session_state.template_fitting_losses = []
                        st.session_state.template_figures = []
                        st.session_state.current_template_figure = 0

                        # process audio files
                        progress_bar = output_container.progress(0, text="Processing ")

                        # fit the direct templates
                        def progress_callback(i, filename):
                            progress_bar.progress(
                                min(0.4 * i / len(audio_files), 1),
                                text=f"Processing {os.path.basename(filename)} "
                                "for Direct Sequencing",
                            )

                        (
                            _,
                            st.session_state.direct_logprobs,
                            st.session_state.direct_figures,
                            st.session_state.direct_cache,
                        ) = sdistill_direct(
                            audio_files,
                            num_orderings=4,
                            progress_callback=progress_callback,
                            cache=st.session_state.direct_cache,
                        )

                        # fit the default template and populate the cache
                        def progress_callback(i, filename):
                            progress_bar.progress(
                                min(0.4 + 0.4 * i / len(audio_files), 1),
                                text=f"Processing {os.path.basename(filename)} "
                                "for Template Sequencing",
                            )

                        _, fitting_loss, figure, st.session_state.template_cache = (
                            sdistill_template(
                                audio_files,
                                progress_callback=progress_callback,
                                cache=st.session_state.template_cache,
                            )
                        )
                        st.session_state.template_fitting_losses.append(fitting_loss)
                        st.session_state.template_figures.append(figure)

                        # fit the other templates
                        for i, template in enumerate(st.session_state.templates):
                            progress_bar.progress(
                                min(
                                    0.8
                                    + 0.2
                                    * (i + 1)
                                    / (len(st.session_state.templates) + 1),
                                    1,
                                ),
                                text=f"Fitting Template {i + 2} of "
                                f"{len(st.session_state.templates) + 1}",
                            )
                            _, fitting_loss, figure, st.session_state.template_cache = (
                                sdistill_template(
                                    audio_files,
                                    template,
                                    cache=st.session_state.template_cache,
                                )
                            )
                            st.session_state.template_fitting_losses.append(
                                fitting_loss
                            )
                            st.session_state.template_figures.append(figure)
                        progress_bar.empty()

                        # organize the figures
                        st.session_state.template_ordering = sorted(
                            range(len(st.session_state.template_fitting_losses)),
                            key=lambda i: st.session_state.template_fitting_losses[i],
                        )
                        st.session_state.template_fitting_losses = [
                            st.session_state.template_fitting_losses[i]
                            for i in st.session_state.template_ordering
                        ]
                        st.session_state.template_figures = [
                            st.session_state.template_figures[i]
                            for i in st.session_state.template_ordering
                        ]

                        # add loss to the figures
                        for figure, logprob in zip(
                            st.session_state.direct_figures,
                            st.session_state.direct_logprobs,
                        ):
                            figure.update_layout(
                                title="Direct Log Probability (higher is better): "
                                f"{logprob:.4f}"
                            )
                        for figure, fitting_loss in zip(
                            st.session_state.template_figures,
                            st.session_state.template_fitting_losses,
                        ):
                            figure.update_layout(
                                title="Template Fitting Loss (lower is better): "
                                f"{fitting_loss:.4f}"
                            )

                        st.rerun()  # force a rerun to update the interface
        else:
            center.text("")  # hack to add a bit of spacing below the button

    # output container
    if len(st.session_state.template_figures) == len(st.session_state.templates) + 1:
        tab1, tab2 = output_container.tabs(["Direct Sequencing", "Template Sequencing"])

        # direct sequencing tab
        col1, col2, col3 = tab1.columns([1, 5, 1])

        for _ in range(13):
            col1.write("")  # hack to center left button vertically
        if col1.button(
            "",
            icon=":material/arrow_back:",
            disabled=st.session_state.current_direct_figure == 0,
            key="direct_back",
        ):
            st.session_state.current_direct_figure = max(
                0,
                st.session_state.current_direct_figure - 1,
            )
            st.rerun()

        col2.plotly_chart(
            st.session_state.direct_figures[st.session_state.current_direct_figure],
            use_container_width=True,
        )

        for _ in range(13):
            col3.write("")  # hack to center right button vertically
        if col3.button(
            "",
            icon=":material/arrow_forward:",
            disabled=st.session_state.current_direct_figure
            == len(st.session_state.direct_figures) - 1,
            key="direct_forward",
        ):
            st.session_state.current_direct_figure = min(
                len(st.session_state.direct_figures) - 1,
                st.session_state.current_direct_figure + 1,
            )
            st.rerun()

        # template ordering tab
        col1, col2, col3 = tab2.columns([1, 5, 1])

        for _ in range(13):
            col1.write("")  # hack to center left button vertically
        if col1.button(
            "",
            icon=":material/arrow_back:",
            disabled=st.session_state.current_template_figure == 0,
            key="template_back",
        ):
            st.session_state.current_template_figure = max(
                0,
                st.session_state.current_template_figure - 1,
            )
            st.rerun()

        col2.plotly_chart(
            st.session_state.template_figures[st.session_state.current_template_figure],
            use_container_width=True,
        )

        for _ in range(13):
            col3.write("")  # hack to center right button vertically
        if col3.button(
            "",
            icon=":material/arrow_forward:",
            disabled=st.session_state.current_template_figure
            == len(st.session_state.template_figures) - 1,
            key="template_forward",
        ):
            st.session_state.current_template_figure = min(
                len(st.session_state.template_figures) - 1,
                st.session_state.current_template_figure + 1,
            )
            st.rerun()


if __name__ == "__main__":
    main()
