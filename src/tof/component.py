# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from abc import abstractmethod

import scipp as sc


class Component:
    kind: str

    @abstractmethod
    def apply(self, neutrons: sc.DataArray) -> sc.DataArray:
        """
        Apply the component to the given neutrons.

        Parameters
        ----------
        neutrons:
            The neutrons to which the component will be applied.

        Returns
        -------
        The modified neutrons.
        """
        raise NotImplementedError
