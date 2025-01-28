{
  description = "Afu study dev environment";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        pythonPackages = python.pkgs;

        regressionLabs = pythonPackages.buildPythonPackage {
          pname = "regression_labs";
          version = "0.1.0";
          format = "setuptools";

          src = pkgs.fetchFromGitHub {
            owner = "osigaud";
            repo = "regressionLabs";
            rev = "main";
            sha256 = "sha256-gdvHs8jFobfWlTN3SbFnpyXx0CiR4UFC+z8dRraR19M=";
          };

          propagatedBuildInputs = with pythonPackages; [
            numpy
            opencv4
          ];

          prePatch = ''
            substituteInPlace setup.py \
              --replace "hash = subprocess.check_output([\"git\", \"rev-parse\", \"HEAD\"], cwd=\".\").decode(\"ascii\").strip()" \
              "hash = \"main\""
          '';

          doCheck = false;
          pythonImportsCheck = ["regression_labs"];
        };

        bbrl = pythonPackages.buildPythonPackage {
          pname = "bbrl";
          version = "0.1.0";
          format = "setuptools";

          src = pkgs.fetchFromGitHub {
            owner = "osigaud";
            repo = "bbrl";
            rev = "master";
            sha256 = "sha256-3u1c4oBbFoAQJjir4/6NSxCmOPQ+OD0jXpGvb7C8EqY=";
          };

          SETUPTOOLS_SCM_PRETEND_VERSION = "0.1.0";

          nativeBuildInputs = with pythonPackages; [
            setuptools-scm
          ];

          propagatedBuildInputs = with pythonPackages; [
            pytorch
            torchvision
            tensorboard
            tqdm
            hydra-core
            numpy
            pandas
            opencv4
            omegaconf
            matplotlib
            seaborn
            scipy
            gymnasium
            moviepy
          ];

          prePatch = ''
            touch src/bbrl/_version.py
            echo '__version__ = "0.1.0"' > src/bbrl/_version.py
          '';

          doCheck = false;
          pythonImportsCheck = ["bbrl"];
        };

        myPythonPackages = with pythonPackages; [
          pytest
          black
          pylint
          moviepy
          numpy
          pre-commit-hooks
          pillow
          matplotlib
          scipy
          seaborn
          omegaconf
          tensorboard
          ipykernel
          opencv4
          jupyter
          torchvision
          pandas
          pydot
          pytorch
          wandb
          hydra-core
          ipympl
          tqdm
          scikit-image
          pyqt6
          qtpy
          gymnasium
          regressionLabs
          bbrl
        ];

        devPackages = with pkgs; [
          just
          (texlive.combine {
            inherit
              (texlive)
              scheme-full
              adjustbox
              collectbox
              tcolorbox
              pgf
              xetex
              ;
          })
          glib
          zlib
          libGL
          fontconfig
          xorg.libX11
          libxkbcommon
          freetype
          dbus

          qt6.full
          qt6.qtbase
          qt6.qtwayland
          libglvnd
        ];

        afu = pythonPackages.buildPythonPackage {
          pname = "afu";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          propagatedBuildInputs = with pythonPackages; [
            pytorch
            bbrl
          ];

          doCheck = false;
          pythonImportsCheck = ["afu"];
        };
        jupyterEnv = python.withPackages (ps: myPythonPackages ++ [afu]);
      in {
        packages = {
          default = jupyterEnv;
        };
        apps.default = {
          type = "app";
          program = "${jupyterEnv}/bin/jupyter-lab";
        };
        devShells.default = pkgs.mkShell {
          buildInputs = myPythonPackages ++ devPackages;

          shellHook = ''
            export QT_QPA_PLATFORM=xcb
            export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt6.qtbase}/lib/qt-${pkgs.qt6.qtbase.version}/plugins"
            export LD_LIBRARY_PATH="${pkgs.qt6.qtbase}/lib:${pkgs.libglvnd}/lib:$LD_LIBRARY_PATH"
          '';
        };
      }
    );
}
