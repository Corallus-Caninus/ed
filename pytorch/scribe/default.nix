{ pkgs ? import <nixpkgs> { config.allowUnfree = true; config.nvidia.acceptLicense = true; } }:


pkgs.mkShell {
  buildInputs = [
#    pkgs.python3
    pkgs.cudaPackages_11.cudatoolkit
    pkgs.linuxPackages.nvidia_x11_legacy470
#    pkgs.glibc
#    pkgs.glib
    pkgs.gcc
#    pkgs.python3Packages.pytorchWithCuda
#    pkgs.python3Packages.transformers
#    pkgs.python3Packages.datasets
#    pkgs.python3Packages.peft
#    pkgs.python3Packages.pip
  ];

  shellHook = ''
    echo "You are now using a NIX environment"
    export CUDA_PATH=${pkgs.cudaPackages_11.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib
    export EXTRA_LD_FLAGS="-L\/lib -L${pkgs.linuxPackages.nvidia_x11_legacy470}\/lib"
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11_legacy470}/lib:${pkgs.cudatoolkit_11}/lib64:$LD_LIBRARY_PATH  
  '';
}

