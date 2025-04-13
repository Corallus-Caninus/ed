{ stdenv, ... }:

stdenv.mkDerivation {
  pname = "ed";
  version = "1.0"; # Set the version of your software

  src = /home/jward/Code/Code_Backup_2024/ed; # Path to your source directory

  buildInputs = [
    pkgs.cudaPackages.cudatoolkit_12
  ]; # Add any build inputs if necessary

  shellHook = ''
    export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit_12}"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
  '';

  buildPhase = ''
    export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit_12}"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    make
  '';

  installPhase = ''
    mkdir -p $out/bin
    cp ed $out/bin/
  '';

}
