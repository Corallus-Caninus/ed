{ pkgs ? import <nixpkgs> { config.allowUnfree = true; config.nvidia.acceptLicense = true; } }:

#TODO: how can we programatically install the virtual environment for the python dependencies?

pkgs.mkShell {
  buildInputs = [
#    pkgs.python3
    pkgs.cudaPackages_11.cudatoolkit
    pkgs.linuxPackages.nvidia_x11_legacy470
#    pkgs.glibc
#    pkgs.glib
    pkgs.gcc11
    pkgs.gcc-unwrapped
    pkgs.ninja
#    pkgs.python3Packages.pytorchWithCuda
#    pkgs.python3Packages.transformers
#    pkgs.python3Packages.datasets
#    pkgs.python3Packages.peft
#    pkgs.python3Packages.pip
  ];

  shellHook = ''
    echo "You are now using a NIX environment"
    export CUDA_PATH=${pkgs.cudaPackages_11.cudatoolkit}
#    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.gcc12}/lib
#    export LD_LIBRARY_PATH=${pkgs.gcc11}/lib
    export EXTRA_LD_FLAGS="-L\/lib -L${pkgs.linuxPackages.nvidia_x11_legacy470}\/lib"
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11_legacy470}/lib:${pkgs.cudatoolkit_11}/lib64:$LD_LIBRARY_PATH  
    alias gcc="${pkgs.gcc12}/bin/gcc"
  '';
}

