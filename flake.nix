{
  description = "torchtyc: static array shape checking for PyTorch via meta tensors";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAll (pkgs: {
        # Everything comes from nixpkgs rather than a uv venv, so this shell
        # works on a NixOS host with no nix-ld.
        default = pkgs.mkShell {
          packages = [
            (pkgs.python313.withPackages (ps: [
              ps.torch
              ps.numpy
              ps.jaxtyping
              ps.einops
              ps.pygls
              ps.watchfiles
              ps.pytest
            ]))
            pkgs.ruff
            pkgs.uv
          ];
          shellHook = ''
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            export TORCHTYC_DEV=1
          '';
        };
      });

      packages = forAll (pkgs: {
        default = pkgs.python313Packages.buildPythonApplication {
          pname = "torchtyc";
          version = "0.1.0";
          pyproject = true;
          src = self;
          build-system = [ pkgs.python313Packages.hatchling ];
          dependencies = with pkgs.python313Packages; [ jaxtyping pygls watchfiles ];
          # The test suite needs torch, which is only in the dev shell.
          doCheck = false;
        };
      });
    };
}
