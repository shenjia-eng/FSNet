"""Two-task GradNorm weighting."""
import copy, math
import torch

VERSION = "fsnet-gradnorm-protected-v2"
DEFAULTS = dict(base_point_coefficient=0.0002, alpha=1.5, weight_lr=0.001,
                positive_floor=0.001, denominator_epsilon=1e-12,
                reference_parameter="fpn.P3_2.weight", warmup_steps=259,
                reference_window_steps=128, ratio_min=0.5, ratio_max=2.0)

def require(value, message):
    if not value:
        raise ValueError(message)

def detached_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: detached_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detached_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detached_copy(v) for v in value)
    return copy.deepcopy(value)

class GradNormBalancer:
    """Own optimizer, isolated from the model/criterion optimizer.
    
    begin() computes a model loss using the OLD detached weights and a separate
    weight-gradient. Call finish_after_model_step() only after a successful
    model optimizer update. Save only between completed updates.
    """
    def __init__(self, device="cpu", config=None, inherited_weights=None):
        self.config = {**DEFAULTS, **(config or {})}
        require(set(self.config) == set(DEFAULTS), "Unknown GradNorm setting")
        for key in ("base_point_coefficient", "weight_lr", "positive_floor", "denominator_epsilon"):
            require(math.isfinite(self.config[key]) and self.config[key] > 0, "Invalid "+key)
        require(math.isfinite(self.config["alpha"]) and self.config["alpha"] >= 0, "Invalid alpha")
        require(type(self.config["warmup_steps"]) is int and self.config["warmup_steps"] >= 0, "Invalid warmup")
        require(type(self.config["reference_window_steps"]) is int and
                0 <= self.config["reference_window_steps"] <= self.config["warmup_steps"], "Invalid reference window")
        require((self.config["warmup_steps"] == 0) == (self.config["reference_window_steps"] == 0), "Warmup/window mismatch")
        require(math.isfinite(self.config["ratio_min"]) and math.isfinite(self.config["ratio_max"]) and
                0 < self.config["ratio_min"] <= 1 <= self.config["ratio_max"], "Invalid weight bounds")
        initial = torch.ones(2, dtype=torch.float32, device=device)
        if inherited_weights is not None:
            initial = torch.as_tensor(inherited_weights, dtype=torch.float32, device=device).clone()
        self.check_bounded_weights(initial)
        self.weights = torch.nn.Parameter(initial)
        self.reference = torch.zeros(2, dtype=torch.float32, device=device)
        self.reference_initialized = False
        self.optimizer = torch.optim.Adam([self.weights], lr=self.config["weight_lr"],
                                          betas=(0.9,0.999), eps=1e-8, weight_decay=0,
                                          foreach=False, fused=False)
        self.model_steps = self.update_steps = self.empty_steps = self.disabled_steps = self.projection_steps = 0
        self.warmup_steps = self.reference_count = self.boundary_steps = 0
        self.reference_sum = torch.zeros(2, dtype=torch.float64, device=device)
        self._pending = None

    @staticmethod
    def check_weights(w):
        require(torch.is_tensor(w) and w.shape == (2,) and w.dtype == torch.float32, "Weight shape/dtype")
        require(bool(torch.isfinite(w).all()) and bool((w > 0).all()), "Weights must be positive finite")
        require(abs(float(w.detach().sum()) - 2.) < 2e-6, "Weights must sum to two")

    def check_bounded_weights(self, w):
        self.check_weights(w)
        ratio = float((w[1]/w[0]).detach())
        require(self.config["ratio_min"]-1e-6 <= ratio <= self.config["ratio_max"]+1e-6, "Weight ratio out of bounds")

    def begin(self, loss_cls, loss_point, shared_parameter, has_ground_truth, adaptive=True):
        require(self._pending is None, "Unfinished model step")
        require(type(has_ground_truth) is bool and type(adaptive) is bool, "Explicit batch flags")
        require(loss_cls.ndim == 0 and loss_point.ndim == 0, "Scalar task losses required")
        q = torch.stack((loss_cls, loss_point*self.config["base_point_coefficient"]))
        require(bool(torch.isfinite(q).all()) and bool((q.detach() >= 0).all()), "Invalid task loss")
        require(q.device == self.weights.device, "Task/weight device mismatch")
        self.check_bounded_weights(self.weights)
        old = self.weights.detach().clone()
        # Deliberately do not expose the trainable weight tensor to model backward.
        total = (old*q).sum()
        record = dict(weights_before=old.cpu().tolist(), raw_loss_cls=float(loss_cls.detach()),
                      raw_loss_point=float(loss_point.detach()), task_losses=q.detach().cpu().tolist(),
                      weighted_losses=(old*q.detach()).cpu().tolist(),
                      effective_point_coefficient_before=self.config["base_point_coefficient"]*float(old[1]/old[0]),
                      empty_batch=not has_ground_truth, adaptive_enabled=adaptive,
                      reference_parameter=self.config["reference_parameter"])
        mode = ("disabled" if not adaptive else
                "warmup" if self.warmup_steps < self.config["warmup_steps"] else
                "empty" if not has_ground_truth else "adaptive")
        record.update(weighting_mode=mode, reference_count_before=self.reference_count,
                      warmup_steps_before=self.warmup_steps,
                      weight_ratio_bounds=[self.config["ratio_min"],self.config["ratio_max"]])
        grad = None
        if mode == "adaptive":
            eps = self.config["denominator_epsilon"]
            require(shared_parameter.requires_grad, "Shared reference is frozen")
            require(bool((q.detach() > eps).all()), "Nonempty task loss too small for GradNorm")
            norms = []
            for task in q:
                g, = torch.autograd.grad(task, (shared_parameter,), retain_graph=True,
                                         create_graph=False, allow_unused=False)
                n = torch.linalg.vector_norm(g.detach())
                require(bool(torch.isfinite(n)) and float(n) > eps, "Shared task gradient missing/zero/nonfinite")
                norms.append(n)
            base_norms = torch.stack(norms).detach()
            if self.reference_initialized:
                ref = self.reference
            elif self.config["warmup_steps"]:
                require(self.reference_count > 0, "No nonempty reference batches in warmup tail")
                ref = (self.reference_sum/self.reference_count).to(dtype=torch.float32)
            else:
                ref = q.detach().clone()
            require(bool((ref > eps).all()), "Invalid reference losses")
            rates = (q.detach()/ref).detach()
            relative = (rates/rates.mean()).detach()
            G = self.weights*base_norms
            targets = (G.detach().mean()*relative.pow(self.config["alpha"])).detach()
            objective = (G-targets).abs().sum()
            grad, = torch.autograd.grad(objective, (self.weights,), retain_graph=False, create_graph=False)
            require(bool(torch.isfinite(grad).all()), "Nonfinite task-weight gradient")
            if not self.reference_initialized:
                self.reference.copy_(ref)
                self.reference_initialized = True
            record.update(base_shared_gradient_norms=base_norms.cpu().tolist(),
                          weighted_shared_gradient_norms=G.detach().cpu().tolist(),
                          target_gradient_norms=targets.cpu().tolist(),
                          relative_training_rates=relative.cpu().tolist(),
                          reference_losses=self.reference.cpu().tolist(),
                          gradnorm_objective=float(objective.detach()),
                          task_weight_gradient=grad.detach().cpu().tolist())
        self._pending = dict(mode=mode, gradient=None if grad is None else grad.detach().clone(), record=record,
                             reference_sample=q.detach().to(torch.float64).clone() if mode=="warmup" and has_ground_truth else None)
        return total, copy.deepcopy(record)

    def finish_after_model_step(self):
        require(self._pending is not None, "No prepared model step")
        pending = self._pending
        projected = bounded = False
        if pending["mode"] == "adaptive":
            self.optimizer.zero_grad(set_to_none=True)
            self.weights.grad = pending["gradient"]
            self.optimizer.step()
            with torch.no_grad():
                require(bool(torch.isfinite(self.weights).all()), "Nonfinite updated weight")
                projected = bool((self.weights < self.config["positive_floor"]).any())
                self.weights.clamp_(min=self.config["positive_floor"])
                self.weights.mul_(2./self.weights.sum())
                lo = 2./(1.+self.config["ratio_max"]); hi = 2./(1.+self.config["ratio_min"])
                bounded = float(self.weights[0]) < lo or float(self.weights[0]) > hi
                self.weights[0].clamp_(min=lo,max=hi)
                self.weights[1].copy_(2.-self.weights[0])
            self.check_bounded_weights(self.weights)
            self.update_steps += 1
            self.projection_steps += int(projected)
            self.boundary_steps += int(bounded)
        elif pending["mode"] == "warmup":
            self.warmup_steps += 1
            sample=pending["reference_sample"]
            if self.warmup_steps > self.config["warmup_steps"]-self.config["reference_window_steps"] and sample is not None:
                require(bool((sample > self.config["denominator_epsilon"]).all()), "Invalid warmup reference batch")
                self.reference_sum.add_(sample); self.reference_count += 1
        elif pending["mode"] == "empty":
            self.empty_steps += 1
        else:
            self.disabled_steps += 1
        self.model_steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        record = dict(pending["record"], weights_after=self.weights.detach().cpu().tolist(),
                      effective_point_coefficient_after=self.config["base_point_coefficient"]*float(self.weights[1].detach()/self.weights[0].detach()),
                      positive_projection=projected, ratio_projection=bounded, model_steps=self.model_steps,
                      warmup_steps=self.warmup_steps, reference_count=self.reference_count, boundary_steps=self.boundary_steps,
                      gradnorm_update_steps=self.update_steps, empty_steps=self.empty_steps,
                      disabled_steps=self.disabled_steps, projection_steps=self.projection_steps)
        self._pending = None
        return record

    def state_dict(self):
        require(self._pending is None, "Cannot checkpoint an unfinished model step")
        return detached_copy(dict(version=VERSION, config=self.config, weights=self.weights,
            reference=self.reference, reference_initialized=self.reference_initialized,
            optimizer=self.optimizer.state_dict(), model_steps=self.model_steps,
            update_steps=self.update_steps, empty_steps=self.empty_steps,
            disabled_steps=self.disabled_steps, projection_steps=self.projection_steps,
            warmup_steps=self.warmup_steps, reference_count=self.reference_count,
            reference_sum=self.reference_sum, boundary_steps=self.boundary_steps))

    def load_state_dict(self, state):
        require(self._pending is None, "Cannot load during a model step")
        required = {"version","config","weights","reference","reference_initialized","optimizer",
                    "model_steps","update_steps","empty_steps","disabled_steps","projection_steps",
                    "warmup_steps","reference_count","reference_sum","boundary_steps"}
        require(set(state)==required and state["version"]==VERSION and state["config"]==self.config, "GradNorm identity/schema")
        self.check_bounded_weights(state["weights"])
        ref=state["reference"]
        require(torch.is_tensor(ref) and ref.shape==(2,) and ref.dtype==torch.float32 and bool(torch.isfinite(ref).all()), "Reference shape/dtype/finite")
        require(type(state["reference_initialized"]) is bool, "Reference initialized flag")
        if state["reference_initialized"]:
            require(bool((ref>self.config["denominator_epsilon"]).all()), "Invalid initialized reference")
        else:
            require(bool((ref==0).all()), "Unexpected uninitialized reference")
        counts=["model_steps","update_steps","empty_steps","disabled_steps","projection_steps","warmup_steps","reference_count","boundary_steps"]
        require(all(type(state[k]) is int and state[k]>=0 for k in counts), "Invalid GradNorm counters")
        require(state["model_steps"]==state["update_steps"]+state["empty_steps"]+state["disabled_steps"]+state["warmup_steps"], "GradNorm counter identity")
        require(state["projection_steps"]<=state["update_steps"], "Projection counter")
        require(state["reference_initialized"]==(state["update_steps"]>0), "Reference/update count mismatch")
        ws=state["warmup_steps"];rc=state["reference_count"];rs=state["reference_sum"]
        require(ws <= self.config["warmup_steps"] and state["boundary_steps"] <= state["update_steps"], "Warmup/boundary count")
        require(not (ws < self.config["warmup_steps"] and (state["update_steps"] or state["empty_steps"])), "Adaptive before warmup")
        require(rc <= max(0,ws-(self.config["warmup_steps"]-self.config["reference_window_steps"])), "Reference sample count")
        require(torch.is_tensor(rs) and rs.shape==(2,) and rs.dtype==torch.float64 and bool(torch.isfinite(rs).all()), "Reference accumulator")
        require(bool((rs==0).all()) if rc==0 else bool((rs>0).all()), "Reference sum/count mismatch")
        if self.config["warmup_steps"] and state["reference_initialized"]:
            require(rc>0 and torch.equal(ref,(rs/rc).to(torch.float32)), "Reference mean mismatch")
        opt=state["optimizer"]
        require(set(opt)=={"state","param_groups"} and len(opt["param_groups"])==1, "Task optimizer schema")
        expected=self.optimizer.state_dict()["param_groups"][0]
        actual=opt["param_groups"][0]
        require(actual==expected, "Task optimizer configuration changed")
        ids=actual["params"];require(len(ids)==1, "Task optimizer cardinality")
        require(set(opt["state"])==({ids[0]} if state["update_steps"] else set()), "Task optimizer states")
        if state["update_steps"]:
            st=opt["state"][ids[0]]
            require(set(st)=={"step","exp_avg","exp_avg_sq"}, "Task Adam state fields")
            for k in ("exp_avg","exp_avg_sq"):
                v=st[k];require(torch.is_tensor(v) and v.dtype==torch.float32 and v.shape==(2,) and bool(torch.isfinite(v).all()), "Task Adam moment invalid")
            require(bool((st["exp_avg_sq"]>=0).all()), "Task Adam variance negative")
            step=st["step"];require(torch.is_tensor(step) and step.numel()==1 and bool(torch.isfinite(step)) and float(step)==state["update_steps"], "Task Adam step/count mismatch")
        # Commit only after validating the complete state.
        with torch.no_grad():
            self.weights.copy_(state["weights"]);self.reference.copy_(ref);self.reference_sum.copy_(rs)
        self.reference_initialized=state["reference_initialized"]
        self.optimizer.load_state_dict(copy.deepcopy(opt))
        for k in counts:setattr(self,k,state[k])

    def start_new_stage(self):
        """New q0 and task Adam, but inherit weights at the selected parent epoch."""
        require(self._pending is None, "Pending parent step")
        return type(self)(device=self.weights.device, config=self.config,
                          inherited_weights=self.weights.detach())
