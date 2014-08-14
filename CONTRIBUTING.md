How to contribute
=================

The preferred way to contribute to pambox is to fork the 
[main repository](http://github.com/achabotl/pambox/) on GitHub:


Getting started
---------------

1. Fork the [project repository](http://github.com/achabotl/pambox): click on the 'Fork' button near the top of the page. This creates a copy of the code under your account on the GitHub server.
2. Clone this copy to your local disk:

    $ git clone git@github.com:YourLogin/pambox.git



Making changes
--------------

* `pambox` uses [git-flow](http://nvie
.com/posts/a-successful-git-branching-model/) as the git branching model.
    * **No commits should be made directly to `master`** 
    * [Install git-flow](https://github.com/nvie/gitflow) and create a `feature` branch like so: `$ git flow feature start <name of your feature>`
* Make commits of logical units.
* Check for unnecessary whitespace with `git diff --check` before committing.
* Make sure you have added the necessary tests for your changes. 
    * Aim for at least 80% coverage on your code
* Run `python setup.py test` to make sure your tests pass
* Run `coverage run --source=pambox setup.py test` if you have the `coverage` 
package installed to generate coverage data
* Check your coverage by running `coverage report`

When you've recorded your changes in Git, then push them to GitHub with:

    $ git push -u origin my-feature

Finally, go to the web page of the your fork of the `pambox` repo,
and click 'Pull request' to send your changes to the maintainers for
review. This will send an email to the committers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  All other tests pass when everything is rebuilt from scratch. On
   Unix-like systems, check with (from the toplevel source folder):

          $ python setup.py test

-  When adding additional functionality, provide at least one
   example script in the ``examples/`` folder. Have a look at other
   examples for reference. 

You can also check for common programming errors with the following
tools:

-  Code with good unittest coverage (at least 80%), check with:

          $ pip install pytest pytest-cov
          $ py.test --cov path/to/pambox

-  No pyflakes warnings, check with:

           $ pip install pyflakes
           $ pyflakes path/to/module.py

-  No PEP8 warnings, check with:

           $ pip install pep8
           $ pep8 path/to/module.py

-  AutoPEP8 can help you fix some of the easy redundant errors:

           $ pip install autopep8
           $ autopep8 path/to/pep8.py
           
           
Style
-----

- Python code should follow the [PEP 8 Style Guide][pep8].
- Python docstrings should follow the [NumPy documentation format][numpydoc].

### Imports

Imports should be one per line.
Imports should be grouped into standard library, third-party,
and intra-library imports. `from` import should follow "regular" `imports`.
Within each group the imports should be alphabetized.
Here's an example:

```python
import sys
from glob import glob

import numpy as np

from pambox.utils import setdbspl
```

Imports of scientific Python libraries should follow these conventions:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
```


Thanks!

[pep8]: http://legacy.python.org/dev/peps/pep-0008/
[numpydoc]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

