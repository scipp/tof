from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


# def deg_to_rad(x):
#     return np.radians(x)


# def rad_to_deg(x):
#     return np.degrees(x)

from .tools import Plot
from . import units


class Chopper:
    def __init__(
        self,
        frequency: float,
        open: Union[List, np.ndarray],
        close: Union[List, np.ndarray],
        distance: float,
        phase: float = 0,
        unit: str = "deg",
        name: str = "",
    ):
        self.frequency = frequency
        self.open = open
        self.close = close
        self.distance = distance
        self.phase = phase
        if unit == "deg":
            self.open = units.deg_to_rad(self.open)
            self.close = units.deg_to_rad(self.close)
            self.phase = units.deg_to_rad(self.phase)
        self.name = name

        self._arrival_times = None
        self._mask = None
        # self.tofs = None

    @property
    def omega(self):
        return 2.0 * np.pi * self.frequency

    @property
    def open_times(self):
        return (self.open + self.phase) / self.omega

    @property
    def close_times(self):
        return (self.close + self.phase) / self.omega

    @property
    def tofs(self):
        return units.s_to_us(self._arrival_times[self._mask])

    def hist(self, bins=100):
        return np.histogram(self.tofs, bins=bins)

    def plot(self, bins=100):
        h, edges = self.hist(bins=bins)
        fig, ax = plt.subplots()
        x = np.concatenate([edges, edges[-1:]])
        y = np.concatenate([[0], h, [0]])
        ax.step(x, y)
        return Plot(fig=fig, ax=ax)

    def __repr__(self):
        return (
            f"Chopper(name={self.name}, distance={self.distance}, "
            f"frequency={self.frequency}, phase={self.phase}, "
            f"cutouts={len(self.open)})"
        )


# class Pulse:
#     def __init__(self, stop: float, start: float = 0):
#         self.start = start
#         self.stop = stop

#     @property
#     def duration(self):
#         return self.stop - self.start


# choppers = dict()

# choppers["TBL1"] = Chopper(
#     frequency=14,
#     openings=np.array([0, 170]),
#     phase=90,
#     distance=8.55,
#     unit="deg",
#     name="TBL1",
# )
# choppers["TBL2"] = Chopper(
#     frequency=14,
#     openings=np.array([0, 10, 90, 260]),
#     phase=20,
#     distance=8.90,
#     unit="deg",
#     name="TBL2",
# )


# def ray_trace(choppers, neutrons, pulse):
#     pass


# def main(choppers, neutrons=1_000_000, pulse_length=2860, detector_position=28.98):
#     # Conversion factors
#     microseconds = 1.0e6
#     v_to_lambda = 3956.0
#     v_to_mev = 437.0

#     pulse = Pulse(stop=pulse_length)

#     ray_trace(choppers=choppers, neutrons=neutrons, pulse_length=pulse_length)

#     # Position of detector
#     # detector_position = 28.98 # 32.4
#     # detector_position = 17  # 28.42  # 32.4
#     # # Monitor
#     # detector_position = 25

#     # # Midpoint between WFM choppers which acts as new source distance for stitched data
#     # wfm_choppers_midpoint = 0.5 * (choppers["TBL1"].distance +
#     #                                choppers["TBL2"].distance)

#     # # Frame colors
#     # colors = ['b', 'k', 'g', 'r', 'cyan', 'magenta']

#     # Make figure
#     fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#     ax.grid(True, color="lightgray", linestyle="dotted")
#     ax.set_axisbelow(True)

#     # Plot the chopper openings
#     for key, ch in choppers.items():
#         dist = [ch.distance, ch.distance]
#         for i in range(0, len(ch.openings), 2):
#             t1 = (ch.openings[i] + ch.phase) / ch.omega * microseconds
#             t2 = (ch.openings[i + 1] + ch.phase) / ch.omega * microseconds
#             ax.plot([t1, t2], dist, color=colors[i // 2])
#         ax.text(t2 + (t2 - t1), ch.distance, ch.name, ha="left", va="center")

#     # Define and draw source pulse
#     x0 = 0.0
#     x1 = pulse_length * microseconds
#     y0 = 0.0
#     y1 = 0.0
#     psize = detector_position / 50.0
#     rect = Rectangle(
#         (x0, y0), x1, -psize, lw=1, fc="orange", ec="k", hatch="////", zorder=10
#     )
#     ax.add_patch(rect)
#     ax.text(x0, -psize, "Source pulse (2.86 ms)", ha="left", va="top", fontsize=6)

#     # Now find frame boundaries and draw frames
#     frame_boundaries = []
#     frame_shifts = []
#     frame_velocities = []

#     # for i in range(nframes):

#     #     # Find the minimum and maximum slopes that are allowed through each frame
#     #     slope_min = 1.0e30
#     #     slope_max = -1.0e30
#     #     for key, ch in choppers.items():

#     #         # For now, ignore Wavelength band double chopper
#     #         if len(ch.openings) == nframes * 2:

#     #             xmin = (ch.openings[i * 2] + ch.phase) / ch.omega * microseconds
#     #             xmax = (ch.openings[i * 2 + 1] +
#     #                     ch.phase) / ch.omega * microseconds
#     #             slope1 = (ch.distance - y1) / (xmin - x1)
#     #             slope2 = (ch.distance - y0) / (xmax - x0)

#     #             if slope_min > slope1:
#     #                 x2 = xmin
#     #                 y2 = ch.distance
#     #                 slope_min = slope1
#     #             if slope_max < slope2:
#     #                 x3 = xmax
#     #                 y3 = ch.distance
#     #                 slope_max = slope2

#     #     # Compute line equation parameters y = a*x + b
#     #     a1 = (y3 - y0) / (x3 - x0)
#     #     a2 = (y2 - y1) / (x2 - x1)
#     #     b1 = y0 - a1 * x0
#     #     b2 = y1 - a2 * x1
#     #     # This is the neutron velocities
#     #     frame_velocities.append([a1 * microseconds, a2 * microseconds])

#     #     y4 = detector_position
#     #     y5 = detector_position

#     #     # This is the frame boundaries
#     #     x5 = (y5 - b1) / a1
#     #     x4 = (y4 - b2) / a2
#     #     frame_boundaries.append([x4, x5])

#     #     # Compute frame shifts from fastest neutrons in frame
#     #     frame_shifts.append((wfm_choppers_midpoint - b2) / a2)

#     #     ax.fill([x0, x1, x4, x5], [y0, y1, y4, y5], alpha=0.3, color=colors[i])
#     #     ax.plot([x0, x5], [y0, y5], color=colors[i], lw=1)
#     #     ax.plot([x1, x4], [y1, y4], color=colors[i], lw=1)
#     #     ax.text(0.5 * (x4 + x5),
#     #             detector_position,
#     #             "Frame {}".format(i + 1),
#     #             ha="center",
#     #             va="top")

#     frame_boundaries = [[0, 100000]]

#     # Plot detector location
#     ax.plot(
#         [0, np.amax(frame_boundaries)],
#         [detector_position, detector_position],
#         lw=3,
#         color="grey",
#     )
#     ax.text(0.0, detector_position, "Detector", va="bottom", ha="left")
#     # Plot WFM choppers mid-point
#     ax.plot(
#         [0, np.amax(frame_boundaries)],
#         [wfm_choppers_midpoint, wfm_choppers_midpoint],
#         lw=1,
#         color="grey",
#         ls="dashed",
#     )
#     ax.text(
#         np.amax(frame_boundaries),
#         wfm_choppers_midpoint,
#         "WFM chopper mid-point",
#         va="bottom",
#         ha="right",
#     )

#     # # Print results as a table:
#     # output = "=================================================================================================\n"
#     # output += "                     "
#     # for i in range(nframes):
#     #     output += "Frame {}      ".format(i + 1)
#     # output += "\nLeft boundary [us]:  "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format(frame_boundaries[i][0])
#     # output += "\nRight boundary [us]: "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format(frame_boundaries[i][1])
#     # output += "\nFrame shift [us]:    "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format(frame_shifts[i])
#     # output += "\nMin speed [m/s]:     "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format(frame_velocities[i][0])
#     # output += "\nMax speed [m/s]:     "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format(frame_velocities[i][1])
#     # output += "\nMin wavelength [AA]: "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format(v_to_lambda / frame_velocities[i][1])
#     # output += "\nMax wavelength [AA]: "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format(v_to_lambda / frame_velocities[i][0])
#     # output += "\nMin energy [meV]:    "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format((frame_velocities[i][0] / v_to_mev)**2)
#     # output += "\nMax energy [meV]:    "
#     # for i in range(nframes):
#     #     output += "{:.5e}  ".format((frame_velocities[i][1] / v_to_mev)**2)
#     # output += "\n================================================================================================="
#     # print(output)

#     # Save the figure
#     ax.set_xlabel("Time [microseconds]")
#     ax.set_ylabel("Distance [m]")
#     fig.savefig("tof_diagram.pdf", bbox_inches="tight")


# if __name__ == "__main__":
#     main()
