{ pkgs ? import <nixpkgs> { config.allowUnfree = true; config.nvidia.acceptLicense = true; } }:


pkgs.mkShell {
  buildInputs = [
    pkgs.python39
    pkgs.pkg-config
    pkgs.cudaPackages_11.cudatoolkit
    pkgs.linuxPackages.nvidia_x11_legacy470
    pkgs.gcc12
    pkgs.gcc-unwrapped
    pkgs.ninja
    pkgs.libxcrypt
  ];

  shellHook = ''
    echo "You are now using a NIX environment"
    export CUDA_PATH=${pkgs.cudaPackages_11.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.gcc12}/lib
    export EXTRA_LD_FLAGS="-L\/lib -L${pkgs.linuxPackages.nvidia_x11_legacy470}\/lib"
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11_legacy470}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.python39}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH="${pkgs.python39}/lib/pkgconfig:$PKG_CONFIG_PATH"
    alias gcc="${pkgs.gcc12}/bin/gcc"
  '';
}
