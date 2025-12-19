import wandb
from wandb import AlertLevel
from trainyourfly.utils.utils import module_to_clean_dict


class WandBLogger:

    def __init__(self, project_name, enabled=True, imgs_every=500):
        self.project_name = project_name
        self.enabled = enabled
        self.log_images_every = imgs_every
        self.initialized = False

    @property
    def run_id(self):
        try:
            return wandb.run.id
        except AttributeError:
            return "NO_RUN_ID"

    def initialize_run(self, config_):
        if self.enabled and not self.initialized:
            model_config = module_to_clean_dict(config_)
            wandb.init(
                project=self.project_name,
                config=model_config,
                group=config_.wandb_group,
            )
            self.initialized = True

    def initialize_sweep(self, sweep_config):
        if self.enabled:
            return wandb.sweep(sweep_config, project=self.project_name)

    def start_agent(self, sweep_id, func):
        if self.enabled:
            wandb.agent(sweep_id, function=func)

    @property
    def sweep_config(self):
        return wandb.config

    def log_metrics(self, epoch, running_loss, total_correct, total, task=None):

        char_ = task if task is not None else ""
        if self.enabled:
            try:
                wandb.log(
                    {
                        "epoch": epoch,
                        f"loss {char_}": running_loss / total,
                        f"accuracy {char_}": total_correct / total,
                    }
                )
            except Exception as e:
                print(f"Error logging running stats to wandb: {e}. Continuing...")

    def log_image(self, vals, name, title, task=None):
        char_ = task if task is not None else ""
        if self.enabled:
            wandb.log(
                {
                    f"{title} image {char_}": wandb.Image(
                        vals, caption=f"{title} image {name}"
                    ),
                }
            )

    def log_dataframe(self, df, title):
        if self.enabled:
            wandb.log({title: wandb.Table(dataframe=df)})

    def update_full_config(self, cfg, protected_keys=("DEVICE",)):
        """Update the live W&B run so that *all* serialisable attributes in
        *cfg* appear in the run's configuration panel.

        Parameters
        ----------
        cfg : object
            A configuration object/module whose attributes hold the training
            parameters.
        protected_keys : iterable[str], optional
            Attribute names that should be skipped because they reference
            complex runtime objects (e.g. *torch.device*) or must not be
            overridden inside an active run.
        """

        if not self.enabled:
            return

        full_cfg = {}
        for k, v in cfg.__dict__.items():
            if k.startswith("__") or k in protected_keys or callable(v):
                continue

            if v is None:
                full_cfg[k] = None
            elif isinstance(v, (int, float, bool, str)):
                full_cfg[k] = v
            elif isinstance(v, (list, tuple)):
                if all(isinstance(item, (int, float, bool, str)) for item in v):
                    full_cfg[k] = list(v)
            # Skip complex, non-serialisable objects (e.g. torch.dtype, torch.device,
            # functions, classes, Path objects, etc.) to avoid corrupting the live
            # config used by training.

        try:
            wandb.config.update(full_cfg, allow_val_change=True)
        except Exception as e:
            print(f"Error updating full config to W&B: {e}. Continuing…")

    def log_validation_stats(
        self, running_loss_, total_correct_, total_, results_, plots, task=None
    ):
        char_ = task if task is not None else ""
        if len(plots) > 0:
            plot_dict = {
                f"Plot {i} {char_}": wandb.Image(plot) for i, plot in enumerate(plots)
            }
        else:
            plot_dict = {}
        if self.enabled:
            wandb.log(
                {
                    f"Validation loss {char_}": running_loss_ / total_,
                    f"Validation accuracy {char_}": total_correct_ / total_,
                    f"Validation results {char_}": wandb.Table(dataframe=results_),
                }
                | plot_dict
            )

    def send_crash(self, message):
        if self.enabled:
            wandb.alert(
                title=f"Error in run at {self.project_name}",
                text=message,
                level=AlertLevel.ERROR,
            )

    def finish(self):
        if self.enabled:
            wandb.finish()
