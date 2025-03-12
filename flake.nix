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

        bbrl = pythonPackages.buildPythonPackage rec {
          pname = "bbrl";
          version = "0.3.3";
          format = "setuptools";

          src = pythonPackages.fetchPypi {
            inherit pname version;
            sha256 = "sha256-6eXoROYw4Zy4Cjc5+YkbrRF6z3q9V1KWYK5LuiA2qqk=";
          };

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

          doCheck = false;
          pythonImportsCheck = ["bbrl"];
        };

        bbrl_gymnasium = pythonPackages.buildPythonPackage {
          pname = "bbrl-gymnasium";
          version = "0.3.7";
          format = "setuptools";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/38/7f/86991c0690b7d14a7ccf29f99167465f444c3157b701cf7adcdfd598383a/bbrl_gymnasium-0.3.7.tar.gz";
            sha256 = "sha256-ojjaJ2TPyYfl5gYQvwfDKqbZjfwvcKAkRztYykWQrEc=";
          };
          propagatedBuildInputs = with pythonPackages; [
            numpy
            gymnasium
            bbrl
            setuptools
          ];
          doCheck = false;
        };

        bbrl_utils = pythonPackages.buildPythonPackage rec {
          pname = "bbrl-utils";
          version = "0.10.3";
          format = "pyproject";

          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/53/d7/e89c7dfb20e862e33a1570ccc81fda5cef3517415c704ab96572af2f9e80/bbrl_utils-0.10.3.tar.gz";
            sha256 = "sha256-mt/IYLbhUqE4jcW/4//Ni8gxufWvHHgkoxiV1PeO0MI=";
          };

          nativeBuildInputs = with pythonPackages; [
            setuptools
            setuptools-scm
          ];

          SETUPTOOLS_SCM_PRETEND_VERSION = version;

          propagatedBuildInputs = with pythonPackages; [
            bbrl
            bbrl_gymnasium
            tensorboard
            pygame
          ];

          preBuild = ''
            export SETUPTOOLS_SCM_PRETEND_VERSION="${version}"
          '';

          doCheck = false;
        };

        mazemdp = pythonPackages.buildPythonPackage rec {
          pname = "mazemdp";
          version = "1.2.7";
          format = "pyproject";

          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/09/15/3aec9e1b63243f180a395ecfe1bdb6e2890bcb1ee13b156c6d9ae24e5434/mazemdp-1.2.7.tar.gz";
            sha256 = "sha256-mXaUZf3mR+FwOSFwlxJ4oPLGyQgdH1RgFRhngciBJvg=";
          };

          nativeBuildInputs = with pythonPackages; [
            setuptools
            setuptools-scm
          ];

          SETUPTOOLS_SCM_PRETEND_VERSION = version;

          propagatedBuildInputs = with pythonPackages; [
            bbrl
            bbrl_gymnasium
            tensorboard
            pygame
            ipyreact
          ];

          preBuild = ''
            export SETUPTOOLS_SCM_PRETEND_VERSION="${version}"
          '';

          doCheck = false;
        };

        ipyreact = pythonPackages.buildPythonPackage rec {
          pname = "ipyreact";
          version = "0.5.0";
          format = "pyproject";

          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-OYs3xXq789RToPtLs005lW8t4hInbeBTYwrZB+q96eU=";
          };

          nativeBuildInputs = with pythonPackages; [
            setuptools
            hatchling
            hatch-vcs
            hatch-jupyter-builder
            jupyterlab
          ];

          propagatedBuildInputs = with pythonPackages; [
            jupyterlab
            jupyter-client
            jupyterlab-widgets
            jupyter-server
            widgetsnbextension
            ipywidgets
          ];

          prePatch = ''
            substituteInPlace pyproject.toml \
              --replace 'jupyterlab==3.*' 'jupyterlab>=3.0.0'
          '';

          doCheck = false;
        };

        tensorboardX = pythonPackages.buildPythonPackage {
          pname = "tensorboardX";
          version = "2.6.2.2";
          format = "setuptools";

          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/02/9b/c2b5aba53f5e27ffcf249fc38485836119638f97d20b978664b15f97c8a6/tensorboardX-2.6.2.2.tar.gz";
            sha256 = "sha256-xkdtfNDVKbC3L0rK2xJp+e2LIvRB6HqE8qO5QLuHtmY=";
          };

          nativeBuildInputs = with pythonPackages; [
            setuptools
            setuptools-scm
          ];

          propagatedBuildInputs = with pythonPackages; [
            numpy
            protobuf
            six
            packaging
          ];

          SETUPTOOLS_SCM_PRETEND_VERSION = "2.6.2.2";

          preBuild = ''
            export SETUPTOOLS_SCM_PRETEND_VERSION="2.6.2.2"
          '';

          doCheck = false;
        };

        afu-rljax = pythonPackages.buildPythonPackage {
          pname = "afu-rljax";
          version = "0.0.6";
          format = "setuptools";

          src = pkgs.fetchFromGitHub {
            owner = "perrin-isir";
            repo = "afu";
            rev = "main";
            sha256 = "sha256-T2SUevL/omsKVrZq1TabAbKZkFd5O5feMoWtUgWIVwI=";
          };

          propagatedBuildInputs = with pythonPackages; [
            jax
            optax
            dm-haiku
            tensorboard
            joblib
            gymnasium
            # pkgs.python312Packages.mujoco
            tensorboardX
            numpy
            pandas
            imageio
            matplotlib
            pkgs.mujoco
          ];

          doCheck = false;
        };

        myPythonPackages = with pythonPackages; [
          pytest
          black
          pylint
          moviepy
          numpy
          pre-commit-hooks
          pygame
          imageio
          imageio-ffmpeg
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
          jax
          optuna
          qtpy
          gymnasium
          pybox2d
          bbrl
          bbrl_utils
          bbrl_gymnasium
          mazemdp
          ipyreact
          afu-rljax
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
          box2d
          fontconfig
          xorg.libX11
          libxkbcommon
          freetype
          dbus
          ffmpeg
          hivemind

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
