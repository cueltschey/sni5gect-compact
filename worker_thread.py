class sni5gect(WorkerThread):
    def start(self):
        self.config.image_name = "ghcr.io/cueltschey/sni5gect-compact"

        self.cleanup_old_containers()
        self.setup_env()
        self.setup_networks()

        self.config.container_volumes[self.config.config_file] = {"bind": "/sni5gect.yaml", "mode": "ro"}
        self.config.container_volumes["/home/charles/collected_iq"] = {"bind": "/iq", "mode": "ro"}
        self.setup_volumes()

        self.start_container()
