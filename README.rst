!`Pip installation <https://github.com/MannLabs/alphastats/workflows/Default%20installation%20and%20tests/badge.svg>`_

**AlphaStats**
==============

An open-source Python package of the AlphaStats ecosystem from the `Mann Labs at the Max Planck Institute of Biochemistry (https://www.biochem.mpg.de/mann). To enable all hyperlinks in this document, please view it at [GitHub <https://github.com/MannLabs/alphatemplate>`_. To enable all hyperlinks in this document, please view it at `GitHub <https://github.com/MannLabs/alphatemplate>`_.


About
------------
An open-source Python package for downstream mass spectrometry data analysis from the `Mann Labs at the Max Planck Institute of Biochemistry <https://www.biochem.mpg.de/mann>`_.


License
------------

AlphaStats was developed by the `Mann Labs at the Max Planck Institute of Biochemistry <https://www.biochem.mpg.de/mann>`_ 
and is freely available with an `Apache License <LICENSE.txt>`_. External Python 
packages (available in the [requirements](requirements) folder) have their own licenses, 
which can be consulted on their respective websites.


Installation
------------

AlphaStats can be installed in an existing Python 3.8 environment with a single `bash` command. *This `bash` command can also be run 
directly from within a Jupyter notebook by prepending it with a `!`*:

.. code-block:: bash
  $ pip install alphastats


Installing AlphaStats like this avoids conflicts when integrating it in other tools, as this does not enforce strict versioning of dependancies. However, if new versions of dependancies are released, they are not guaranteed to be fully compatible with AlphaStats. While this should only occur in rare cases where dependencies are not backwards compatible, you can always force AlphaStats to use dependancy versions which are known to be compatible with:

.. code-block:: bash
  $ pip install "alphastats[stable]"

NOTE: You might need to run `pip install pip==21.0` before installing AlphaStats like this. Also note the double quotes `"`.

For those who are really adventurous, it is also possible to directly install any branch (e.g. `@development`) with any extras (e.g. `#egg=alphastats[stable,development-stable]`) from GitHub with e.g.

.. code-block:: bash
  $ pip install "git+https://github.com/MannLabs/alphastats.git@development#egg=alphastats[stable,development-stable]"



Usage
------------



Python and Jupyter Notebook
------------------------

AlphaStats can be imported as a Python package into any Python script or notebook with the command `import alphastats`.


Troubleshooting
------------

In case of issues, check out the following:

* `Issues <https://github.com/MannLabs/alphastats/issues>`_: Try a few different search terms to find out if a similar problem has been encountered before

* `Discussions <https://github.com/MannLabs/alphastats/discussions>`_: Check if your problem or feature requests has been discussed before.


Citations
------------

There are currently no plans to draft a manuscript.


How to contribute
------------------------


If you like this software, you can give us a `star <https://github.com/MannLabs/alphastats/stargazers>`_ to boost our visibility! 
All direct contributions are also welcome. Feel free to post a new `issue <https://github.com/MannLabs/alphastats/issues>`_ or clone the 
repository and create a `pull request <https://github.com/MannLabs/alphastats/pulls>`_ with a new branch. For an even more interactive 
participation, check out the `discussions <https://github.com/MannLabs/alphastats/discussions>`_ and the 
the `Contributors License Agreement <misc/CLA.md>`_  to boost our visibility! All direct contributions are also welcome. 


Changelog
------------

See the `HISTORY.md <HISTORY.md>`_ for a full overview of the changes made in each version.

