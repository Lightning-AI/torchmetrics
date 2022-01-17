from pprint import pprint

import fire


class AssistantCLI:
    @staticmethod
    def prune_packages(req_file: str, *pkgs: str) -> None:
        """Prune packages from requirement file."""
        with open(req_file) as fp:
            lines = fp.readlines()

        for pkg in pkgs:
            lines = [ln for ln in lines if not ln.startswith(pkg)]
        pprint(lines)

        with open(req_file, "w") as fp:
            fp.writelines(lines)


if __name__ == "__main__":
    fire.Fire(AssistantCLI)
