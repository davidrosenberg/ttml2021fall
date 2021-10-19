
# Setup with Pipenv

Step 1: Install Pipenv: 
* Run `$ pip install pipenv` on your command prompt/terminal, OR
* Run `$ pip3 install pipenv` if the previous command doesn't work


Step 2: Create virtual environment 
* Navigate to the directory with Pipfile and/or Pipfile.lock in command prompt/terminal
* Run `$ pipenv sync` which will create a virtual environment and install all the python
packages with the "locked" versions from the Pipfile.lock.

Step 3: Enter virtual environment
* Once packages are finished installing, run `$ pipenv shell`.
* Alternative: instead of going into the shell, you can also always run `$ pipenv run <command>`
 (e.g. `pipenv run python3 example_module.py`)

Step 4: Run your Jupyter Notebook
* Either (1) start your jupyter notebook server like normal *if you are in the pipenv
shell* or (2) run `$ pipenv run jupyter notebook`.
* If you want to add a package while running a Jupyter notebook, you don't have to stop the
jupyter server. Instead, open a **new** terminal/command line prompt and navigate to the Pipfile.
If you run, `$ pipenv install example-package` then `example-package` will now be available to your
jupyter notebook. This only works if in the shell and does **not** work with the `$ pipenv run <command>`
variant.

Other useful notes/tips:
* The Pipfile is the human readable list of python packages and is the one that can be
directly edited if we want to type package names/versions/options there (much like a requirements.txt).
The Pipfile.lock is the machine readable version which locks all the version of the packages that you are
currently using (even if you did not specify one). This allows for anyone to run your python
code with the correct versions. 
* If you want to install a new package in the pipenv virtual environment, simply run 
`$ pipenv install example-package`. You can then check your Pipfile and see that the package 
automatically added. 
* If you are creating a new project, you can run `$ pipenv install` or `$ pipenv install package1 package2 package3`
 in repository to automatically create a Pipfile and/or Pipfile.lock
* Sometimes `pipenv install` (which basically re-installs/updates all your packages listed in the Pipfile)
will take a long time to lock. This is not necessary, for example, when you are first starting a project
and need to keep adding packages one by one. In this case, you can run `pipenv install example-package --skip-lock`,
which will still correctly install and add it to your Pipfile but without the long wait. Once you are
ready to lock the dependencies, you can run `pipenv lock`
* Pipenv can be annoying sometimes. If having a nonsensical problem, this almost always works:
     1. Run `$ pipenv --rm` which deletes the virtual environment  
     2. Delete Pipfile.lock: `$ rm Pipfile.lock`
     3. Run `$ pipenv install`
