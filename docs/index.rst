.. raw:: html

   <div style="display: block; margin-left: auto; margin-right: auto; width: 60%;">
      <img src="_static/logo.svg" width="80%" />
   </div>
   <style> .transparent {opacity:0; font-size:16px} </style>

.. role:: transparent

:transparent:`Tof`
******************

A simple tool to create time-of-flight chopper cascade diagrams.

**Scope:**

* ``tof`` is designed to be simple and lightweight, it will never replace `McStas <https://www.mcstas.org/>`_
* it assumes all neutrons travel in a straight line
* it does not simulate absorption or scattering effects

Installation
============

You can install from ``pip`` using

.. code-block:: sh

   pip install tof

You can install from ``conda`` using

.. code-block:: sh

   conda install -c conda-forge -c scipp tof

.. toctree::
   :maxdepth: 2
   :hidden:

   short-example
   sources
   components
   multiple-pulses
   wfm
   ess/index
   api
   Release notes <https://github.com/scipp/tof/releases>
