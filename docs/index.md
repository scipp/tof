:::{image} _static/logo.svg
:class: only-light
:alt: Tof
:width: 60%
:align: center
:::
:::{image} _static/logo-dark.svg
:class: only-dark
:alt: Tof
:width: 60%
:align: center
:::

```{raw} html
   <style>
    .transparent {display: none; visibility: hidden;}
    .transparent + a.headerlink {display: none; visibility: hidden;}
   </style>
```

```{role} transparent
```

# {transparent}`Tof`

<div style="font-size:1.2em;font-style:italic;color:var(--pst-color-text-muted);text-align:center;">
  A simple tool to create time-of-flight chopper cascade diagrams
  </br></br>
</div>

## ✨ [Try it in your browser](https://scipp.github.io/toflite/lab/index.html?path=app.ipynb) ✨


## Installation

To install Tof and all of its dependencies, use

`````{tab-set}
````{tab-item} pip
```sh
pip install tof
```
````
````{tab-item} conda
```sh
conda install -c conda-forge -c scipp tof
```
````
`````

## Get in touch

- If you have questions that are not answered by these documentation pages, ask on [discussions](https://github.com/scipp/tof/discussions). Please include a self-contained reproducible example if possible.
- Report bugs (including unclear, missing, or wrong documentation!), suggest features or view the source code [on GitHub](https://github.com/scipp/tof).

```{toctree}
---
hidden:
---

demo
sources
components
multiple-pulses
wfm
dashboard
ess/index
api-reference/index
developer/index
about/index
```
