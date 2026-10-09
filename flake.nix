{
  description = "uv dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          uv
          stdenv.cc
          pkg-config
          cmake
          ninja
          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cudart
        ];

        shellHook = ''
          export CUDA_HOME=${pkgs.cudaPackages.cuda_nvcc}
          export CUDA_PATH=${pkgs.cudaPackages.cuda_nvcc}

          export CUDA_INCLUDE_DIR=${pkgs.lib.getDev pkgs.cudaPackages.cuda_cudart}/include
          export CUDA_LIB_DIR=${pkgs.lib.getLib pkgs.cudaPackages.cuda_cudart}/lib

          export PATH=${pkgs.lib.getBin pkgs.cudaPackages.cuda_nvcc}/bin:$PATH
          export CPATH=$CUDA_INCLUDE_DIR:$CPATH
          export LIBRARY_PATH=$CUDA_LIB_DIR:$LIBRARY_PATH
          export LD_LIBRARY_PATH=$CUDA_LIB_DIR:$LD_LIBRARY_PATH

          export CC=${pkgs.stdenv.cc}/bin/cc
          export CXX=${pkgs.stdenv.cc}/bin/c++
          export CUDAHOSTCXX=${pkgs.stdenv.cc}/bin/c++

          echo "CC=$CC"
          echo "CXX=$CXX"
          echo "nvcc=$(command -v nvcc)"
          echo "g++=$(command -v g++ || true)"
          echo "c++=$(command -v c++ || true)"
        '';
      };
    };
}
