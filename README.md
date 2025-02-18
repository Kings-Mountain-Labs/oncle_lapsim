# oncle-lapsim
A lap time simulation and vehicle dynamics toolbox made by people who like Oncle Jazz by Men I Trust

This project originates from Arash Mehdipour's[^1] Matlab implementation of Chris Patton's 2013 PhD dissertation[^2], which I (Ian Rist) [^1] then ported to python and greatly extended. This toolbox also includes a Magic-Formula Tire model which is based on the Matlab implementation by Marco Furlan [^3].

[^1]: Spartan Racing, San Jose State University Formula SAE

[^2]: [Development of vehicle dynamics tools for motorsports](ir.library.oregonstate.edu/concern/graduate_thesis_or_dissertations/tx31qm51z)

[^3]: [MFeval](mfeval.wordpress.com)

# Getting Started
This getting started guide uses the Github CLI to clone the repository because we have tended to find that it is the easiest way to get new git users correctly configured.

## Installation  <span style="color:red">FOLLOW THIS IN ORDER</span>
1. Dependencies
    - [Install VS Code](https://code.visualstudio.com/download)
    - [Install Git for Windows](https://git-scm.com/downloads)
    - [Install the Github CLI](https://cli.github.com/)
    - [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
    - [Install Rust Toolchain](https://www.rust-lang.org/tools/install) <span style="color:red">ON WINDOWS, IT WILL ASK YOU IF YOU WANT TO INSTALL MSVC, YOU DO WANT TO</span>
    - Restart your computer for good measure
2. Clone the repository
    - Open the github website to our repository
    - Click the green code button, click Github CLI, and copy the command
    - Open the windows terminal (or powershell)
    - Navigate to the folder you want to clone to (`cd C:\Users\username\Documents\` for instance and `cd ..` to back one)
    - Paste and run the command you copied
    - Select the folder you want to clone to and clone
    - Now `cd oncle_lapsim` to get into the repository
    - And `code .` to open VS Code in the repository
3. Install the dependencies
    - Open a terminal (in VS Code) in the folder you cloned to
    ```console
        uv sync
    ```
    - The first time you do this you will invalidate your current terminal (vs-code will make it orange) and you just need to make a new terminal
    - Now, in vscode, press Cmd+Shift+P and type "Python: Select Interpreter" and select the virtual environment in (.venv)
    
    - Building the Tire Model Improvements (not needed for the simulation to run, but causes dramatic performance improvements)
        - Create a new terminal in VS Code
        - Change directories to the folder with the tire model lib
    ```console
    cd toolkit\pacejka_rs\
    ```
        - Compile the Rust libraries (this will take a while the first time and you will need to do it whenever I have updated the rust code)
    ```console
    maturin develop --release
    ```
        - You should see a bunch of stuff about compiling and linking and then it should say `Build succeeded, installing package`
        - Now you can close the terminal