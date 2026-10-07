def input_digest(batch: dict, patient_by_path: dict[str, str],
                 source_by_path: dict[str, str], eligibility: dict,
                 classes: list[str]) -> tuple[str, str, str, str]:
    paths = [str(path) for path in batch["im_file"]]
    try:
        patient_groups = [patient_by_path[path] for path in paths]
        eligibility_rows = [[eligibility[source_by_path[path]][name] for name in classes]
                            for path in paths]
    except KeyError as exc:
        raise RuntimeError(f"Training path has no frozen patient/eligibility mapping: {exc}") from exc
    ordered_paths = json.dumps(paths, ensure_ascii=False, separators=(",", ":")).encode()
    ordered_patients = json.dumps(patient_groups, ensure_ascii=False, separators=(",", ":")).encode()
    ordered_eligibility = json.dumps(eligibility_rows, ensure_ascii=False, separators=(",", ":")).encode()
    path_hash = hashlib.sha256(ordered_paths).hexdigest()
    patient_hash = hashlib.sha256(ordered_patients).hexdigest()
    eligibility_hash = hashlib.sha256(ordered_eligibility).hexdigest()
    digest = hashlib.sha256()
    digest.update(ordered_paths)
    digest.update(ordered_patients)
    digest.update(ordered_eligibility)
    for key in ("img", "batch_idx", "cls", "bboxes"):
        tensor = batch[key].detach().contiguous().cpu()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), path_hash, patient_hash, eligibility_hash



def p1() -> None:
    if read_json(ROOT / "source_hashes.json")["status"] != "PASS":
        raise RuntimeError("P0 has not passed")
    value = protocol()
    pred = torch.tensor([[.1, .1, .8, .8]], dtype=torch.float64, requires_grad=True)
    gt = torch.tensor([[0., 0., 1., 1.]], dtype=torch.float64)
    aligned_iou, aligned_d2 = aligned_geometry(pred, gt)
    if not torch.allclose(aligned_iou, torch.tensor([.49], dtype=torch.float64), atol=1e-12):
        raise RuntimeError("Aligned geometry IoU sanity failed")
    iou = torch.tensor([.5, .75], dtype=torch.float64, requires_grad=True)
    d2 = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    gt_axis = torch.tensor([[100., 200., 200., 240.]], dtype=torch.float32)
    pred_axis = torch.tensor([[110., 220., 210., 260.]], dtype=torch.float32,
                             requires_grad=True)
    stride_axis = torch.tensor([8.], dtype=torch.float32)
    center = axis_center_term(pred_axis, gt_axis, stride_axis)
    if not torch.allclose(center, torch.tensor([.3]), rtol=0, atol=1e-6):
        raise RuntimeError("Frozen axis center 0.3 toy failed")
    edge_grad = torch.autograd.grad(center.sum(), pred_axis)[0]
    if not torch.allclose(edge_grad, torch.tensor([[.0025, .00625, .0025, .00625]]),
                          rtol=0, atol=1e-5):
        raise RuntimeError("Frozen axis center coordinate gradient failed")
    small_gt = torch.tensor([[20., 30., 24., 36.]])
    small_pred = torch.tensor([[22., 34., 26., 40.]])
    for stride_value, expected in ((8., .375), (16., .1875)):
        observed = axis_center_term(small_pred, small_gt, torch.tensor([stride_value]))
        if not torch.allclose(observed, torch.tensor([expected]), rtol=0, atol=1e-6):
            raise RuntimeError("Frozen small-GT stride-floor toy failed")
    for bad_gt in (torch.tensor([[24., 30., 20., 36.]]),
                   torch.tensor([[20., 30., 20., 36.]])):
        try:
            axis_center_term(small_pred, bad_gt, torch.tensor([8.]))
        except ValueError:
            pass
        else:
            raise RuntimeError("Invalid assigned GT was silently accepted")
    gt_with_grad = gt_axis.clone().requires_grad_(True)
    stride_with_grad = stride_axis.clone().requires_grad_(True)
    detach_probe = axis_center_term(pred_axis, gt_with_grad, stride_with_grad)
    det_grads = torch.autograd.grad(detach_probe.sum(), (gt_with_grad, stride_with_grad),
                                    allow_unused=True)
    if any(g is not None for g in det_grads):
        raise RuntimeError("Axis center failed to detach assigned GT or native stride")
    axis_loss = assigned_geometry_loss(iou, d2, pred, "axis_center_v1",
                                       torch.tensor([.3, .375], dtype=torch.float64))
    expected_axis = (((2 - 2 ** .5) + .3) + ((2 - 2 ** .75) + .375)) / 2
    if not np.isclose(float(axis_loss.detach()), expected_axis, rtol=0, atol=1e-12):
        raise RuntimeError("Frozen axis total loss failed")
    grad_iou = torch.autograd.grad(axis_loss, iou)[0]
    expected_iou_grad = torch.tensor([-.5 * np.log(2) * (2 ** .5),
                                      -.5 * np.log(2) * (2 ** .75)], dtype=torch.float64)
    if not torch.allclose(grad_iou, expected_iou_grad, rtol=0, atol=1e-12):
        raise RuntimeError("Original overlap gradient changed")
    dtype_sanity = {}
    for dtype in (torch.float32, torch.bfloat16):
        sample_pred = pred_axis.detach().to(dtype).requires_grad_(True)
        sample_gt = gt_axis.to(dtype)
        sample_center = axis_center_term(sample_pred, sample_gt, stride_axis.to(dtype))
        sample_grad = torch.autograd.grad(sample_center.sum(), sample_pred)[0]
        if not bool(torch.isfinite(sample_center).all()) or not bool(torch.isfinite(sample_grad).all()):
            raise RuntimeError(f"Frozen axis {dtype} finite/gradient sanity failed")
        dtype_sanity[str(dtype)] = {"center": float(sample_center[0]),
                                    "center_dtype": str(sample_center.dtype),
                                    "gradient_finite": True}
    original = assigned_geometry_loss(iou, d2, pred, "original")
    expected_original = (((2 - 2 ** .5) + 1) + ((2 - 2 ** .75) + 2)) / 2
    if not np.isclose(float(original.detach()), expected_original, rtol=0, atol=1e-12):
        raise RuntimeError("Production original geometry identity failed")
    empty = assigned_geometry_loss(iou[:0], d2[:0], pred, "axis_center_v1")
    if empty.item() != 0 or not empty.requires_grad:
        raise RuntimeError("Production connected empty-positive zero failed")
    single = assigned_geometry_loss(iou[:1], d2[:1], pred, "axis_center_v1",
                                    torch.tensor([.3], dtype=torch.float64))
    if not torch.allclose(single, ((2.0 - 2.0 ** iou[:1]) + .3).mean(), atol=1e-12):
        raise RuntimeError("Single-positive axis loss failed")
    empty.backward()
    if pred.grad is None or not torch.equal(pred.grad, torch.zeros_like(pred)):
        raise RuntimeError("Empty-positive zero lost box graph connection")
    original_loss_source = (Path(value["pilot"]["source_root"]) / "v6a2_loss.py").read_text()
    pilot_loss_source = (ROOT / "v6a2_loss.py").read_text()
    start = "        # Reuse the exact same native assigner only to obtain target_scores."
    end = "        # Native v8DetectionLoss.loss multiplies its components by batch size."
    if (original_loss_source.split(start, 1)[1].split(end, 1)[0] !=
            pilot_loss_source.split(start, 1)[1].split(end, 1)[0]):
        raise RuntimeError("Native assignment/Q block changed in pilot loss")
    toy_ap = np.tile(np.arange(10, dtype=np.float64) / 100, (6, 1))
    toy = summarize_ap_matrix(list(range(6)), toy_ap, [.50 + .05 * j for j in range(10)],
                              0.0, float(toy_ap.mean()), value["class_names"])
    if (not np.isclose(toy["H"], float(toy_ap[:, 5:].mean())) or
            not np.isclose(toy["macro_ap"]["0.95"], .09)):
        raise RuntimeError("Per-IoU AP export sanity failed")
    toy_batch = {"im_file": ["toy"], "img": torch.zeros((1, 3, 2, 2)),
                 "batch_idx": torch.zeros(1), "cls": torch.zeros(1),
                 "bboxes": torch.zeros((1, 4))}
    toy_digest_e = input_digest(toy_batch, {"toy": "patient"}, {"toy": "source"},
                                {"source": {"class": "E"}}, ["class"])
    toy_digest_u = input_digest(toy_batch, {"toy": "patient"}, {"toy": "source"},
                                {"source": {"class": "U"}}, ["class"])
    if (toy_digest_e[0] == toy_digest_u[0] or toy_digest_e[3] == toy_digest_u[3] or
            toy_digest_e[1:3] != toy_digest_u[1:3]):
        raise RuntimeError("Per-step digest does not bind eligibility independently of patient/image IDs")
    model, criterion = model_and_loss(value, torch.device("cpu"), "box_geo")
    _load_checkpoint(Path(value["m3_checkpoint"]), model,
                     Path(value["pilot"]["source_root"]) / "protocol.m3_b60w30.frozen.json")
    with torch.no_grad():
        example_raw = model(torch.zeros(1, 3, 128, 128))["one2one"]
    anchors, native_stride = make_anchors(example_raw["feats"],
                                           criterion.native.stride, 0.5)
    nslots = example_raw["boxes"].shape[-1]
    if (native_stride.shape != (nslots, 1) or anchors.shape != (nslots, 2) or
            set(native_stride[:, 0].tolist()) != {8.0, 16.0, 32.0}):
        raise RuntimeError("Native slot stride does not align to raw slot axis")
    synthetic_fg = torch.zeros((1, nslots), dtype=torch.bool)
    selected = []
    for scale in (8.0, 16.0, 32.0):
        index = int(torch.where(native_stride[:, 0] == scale)[0][0])
        synthetic_fg[0, index] = True
        selected.append(scale)
    aligned_stride = native_stride[:, 0].unsqueeze(0).expand(1, -1)[synthetic_fg]
    if aligned_stride.tolist() != selected:
        raise RuntimeError("Foreground mask selected the wrong native slot stride")
    sample_anchor = torch.tensor([[64.0, 64.0]], dtype=torch.float32)
    sample_stride = torch.tensor([[8.0]], dtype=torch.float32)
    raw_bf16 = torch.tensor([[[1.0, 1.0, 1.03125, 1.0]]],
                            dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        decoded = axis_decoded_pixel_boxes(criterion.native, sample_anchor,
                                            raw_bf16, sample_stride)[0]
    near_gt = torch.tensor([[504.0, 504.0, 520.0, 520.0]], dtype=torch.float32)
    near_center = axis_center_term(decoded, near_gt, sample_stride[:, 0])
    near_grad = torch.autograd.grad(near_center.sum(), raw_bf16)[0]
    if (decoded.dtype != torch.float32 or
            not torch.allclose(near_center, torch.tensor([.00390625]), rtol=0, atol=1e-6) or
            not torch.allclose(near_grad[0, 0, 2].float(), torch.tensor(.125),
                               rtol=0, atol=1e-5)):
        raise RuntimeError("BF16 native raw-box FP32 decode lost a small center residual")
    if criterion.geo_variant != "original":
        raise RuntimeError("Default geometry path no longer original")
    set_trainable(model, "box_geo")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    if not trainable or any("one2one_cv2" not in n and "v6a2_quality_heads" not in n
                           for n in trainable):
        raise RuntimeError("Trainable parameter set changed")
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            expected_training = any(module is m for m in model.model[-1].one2one_cv2.modules())
            if module.training != expected_training:
                raise RuntimeError("BN state differs from original box mode")
    del model, criterion
    atomic_json(ROOT / "sanity.json", {"status": "PASS", "iou": iou.detach().tolist(),
                                      "d2": d2.detach().tolist(),
                                      "original_geo": float(original.detach()),
                                      "axis_geo": float(axis_loss.detach()),
                                      "axis_center_toy": float(center.detach()),
                                      "dtype_sanity": dtype_sanity,
                                      "trainable_parameter_count": len(trainable),
                                      "test_data_read": 0})



def train(arm: str, seed: int) -> None:
    if read_json(ROOT / "source_hashes.json")["status"] != "PASS" or \
            read_json(ROOT / "sanity.json")["status"] != "PASS":
        raise RuntimeError("P0/P1 have not passed")
    value = protocol()
    torch.backends.cudnn.benchmark = False
    verify_sources(value)
    value["seed"] = seed
    if socket.gethostname() != "hpc6" or os.environ.get("CUDA_VISIBLE_DEVICES") != value["pilot"]["gpu_uuid"]:
        raise RuntimeError("Pilot GPU identity changed after P0")
    observed = read_json(ROOT / "source_hashes.json")["observed_sha256"]
    if sha(MANIFEST) != observed[str(MANIFEST)]:
        raise RuntimeError("P0 frozen source manifest changed")
    for path_string in (value["m3_checkpoint"], value["train_list"],
                        value["development_validation_list"]):
        if sha(Path(path_string)) != observed[path_string]:
            raise RuntimeError(f"Frozen pilot input changed after P0: {path_string}")
    mode, variant = ARMS[arm]
    population = read_json(Path(value["pilot"]["population_json"]))
    patient_by_path = {str(row["path"]): str(row["group"]) for row in population["train"]}
    if len(patient_by_path) != 14934:
        raise RuntimeError("Frozen training patient mapping is incomplete")
    reference_arms = () if arm == "B2" else ("B2",)
    reference_steps = {}
    for previous_arm in reference_arms:
        previous_dir = ROOT / "runs" / f"seed{seed}" / previous_arm
        previous_status = read_json(previous_dir / "status.json")
        if previous_status["status"] != "PASS" or previous_status["total_optimizer_steps"] != 4150:
            raise RuntimeError(f"Reference arm {previous_arm} is incomplete")
        rows = [json.loads(line) for line in (previous_dir / "input_hashes.jsonl").read_text().splitlines()]
        if len(rows) != 4150:
            raise RuntimeError(f"Reference arm {previous_arm} has {len(rows)} input steps")
        reference_steps[previous_arm] = rows
    run_dir = ROOT / "runs" / f"seed{seed}" / arm
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    device = torch.device("cuda:0")
    loader = make_loader(value, Path(value["train_list"]), augment=True,
                         batch=180, workers=40, shuffle=True, seed=seed, pin_memory=True)
    model, criterion = model_and_loss(value, device, mode)
    criterion.geo_variant = variant
    _load_checkpoint(Path(value["m3_checkpoint"]), model,
                     Path(value["pilot"]["source_root"]) / "protocol.m3_b60w30.frozen.json")
    head = model.model[-1]
    box_params = list(head.one2one_cv2.parameters())
    q_params = list(head.v6a2_quality_heads.parameters())
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    if (not box_params or not q_params or
            any("one2one_cv2" not in n and "v6a2_quality_heads" not in n for n in trainable)):
        raise RuntimeError("Unexpected trainable parameters")
    initial_box_hash = _parameter_hash(head.one2one_cv2.state_dict())
    initial_q_hash = _parameter_hash(head.v6a2_quality_heads.state_dict())
    frozen_initial = frozen_hash(model)
    optimizer = torch.optim.AdamW([
        {"params": box_params, "lr": value["box_optimizer"]["box_lr"]},
        {"params": q_params, "lr": value["box_optimizer"]["q_lr"]}],
        weight_decay=value["box_optimizer"]["weight_decay"])
    status = {"status": "RUNNING", "arm": arm, "mode": mode, "geo_variant": variant,
              "initial_box_hash": initial_box_hash, "initial_q_hash": initial_q_hash,
              "source_m3_sha256": EXPECTED_M3_SHA, "protocol_sha256": sha(PROTOCOL),
              "frozen_protocol_sha256": sha(FROZEN), "decision_rule_sha256": sha(RULE),
              "seed": seed, "epochs": [], "test_data_read": 0}
    atomic_json(run_dir / "status.json", status)
    total_steps = 0
    started = time.monotonic()
    try:
        with (run_dir / "input_hashes.jsonl").open("w") as hashes, \
             (run_dir / "train_metrics.jsonl").open("w") as metrics:
            for epoch in range(1, 51):
                model.train()
                set_trainable(model, mode)
                count = 0
                steps = 0
                for step, data in enumerate(loader, 1):
                    input_sha, paths_sha, patients_sha, eligibility_sha = input_digest(
                        data, patient_by_path, criterion.sources, criterion.matrix, criterion.classes)
                    actual_bs = len(data["im_file"])
                    current_key = (epoch, step, input_sha, paths_sha, patients_sha,
                                   eligibility_sha, actual_bs)
                    for previous_arm, rows in reference_steps.items():
                        if total_steps >= len(rows):
                            raise RuntimeError(f"{arm} has more steps than {previous_arm}")
                        prior = rows[total_steps]
                        prior_key = (prior["epoch"], prior["step"], prior["input_sha256"],
                                     prior["ordered_paths_sha256"],
                                     prior["ordered_patient_groups_sha256"],
                                     prior["ordered_eligibility_sha256"], prior["actual_batch_size"])
                        if current_key != prior_key:
                            raise RuntimeError(f"Paired inputs differ from {previous_arm} at optimizer step {total_steps + 1}")
                    batch = move_batch(data, device)
                    optimizer.zero_grad(set_to_none=True)
                    with amp_context(value, device):
                        loss, audit = criterion(model(batch["img"]), batch, mode,
                                                lambda_geo=(value["lambda_geo"] * actual_bs
                                                            if mode == "box_geo" else 0.0),
                                                epoch=epoch)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"Nonfinite loss E{epoch} S{step}")
                    loss.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                           for p in box_params + q_params):
                        raise RuntimeError(f"Nonfinite gradient E{epoch} S{step}")
                    if any(p.grad is not None for n,p in model.named_parameters() if "one2one_cv2" not in n and "v6a2_quality_heads" not in n):
                        raise RuntimeError("Gradient reached a frozen parameter")
                    optimizer.step()
                    if any(not bool(torch.isfinite(p).all()) for p in box_params + q_params):
                        raise RuntimeError("Nonfinite parameter after optimizer step")
                    total_steps += 1
                    steps += 1
                    count += actual_bs
                    hashes.write(json.dumps({"seed": seed, "arm": arm, "epoch": epoch, "step": step,
                                             "input_sha256": input_sha,
                                             "ordered_paths_sha256": paths_sha,
                                             "ordered_patient_groups_sha256": patients_sha,
                                             "ordered_eligibility_sha256": eligibility_sha,
                                             "actual_batch_size": actual_bs}) + "\n")
                    metrics.write(json.dumps({"epoch": epoch, "step": step,
                                              "optimizer_step": total_steps,
                                              "loss_finite": True,
                                              "gradients_finite": True,
                                              "loss": float(loss.detach()),
                                              "native_box": audit["native_box"],
                                              "q": audit["q"], "geo": audit["geo"],
                                              "fg_count": audit["fg_count"],
                                              "geo_stats": audit["geo_stats"],
                                              "actual_batch_size": actual_bs,
                                              "effective_lambda_geo": value["lambda_geo"] * actual_bs
                                              if mode == "box_geo" else 0.0}) + "\n")
                    if step % 10 == 0:
                        hashes.flush()
                        metrics.flush()
                if count != 14934 or steps != 83:
                    raise RuntimeError(f"Incomplete E{epoch}: {count} images, {steps} steps")
                status["epochs"].append({"epoch": epoch, "images": count,
                                         "steps": steps, "total_steps": total_steps})
                atomic_json(run_dir / "status.json", status)
                verify_freeze(model, frozen_initial)
                if epoch in (10,20,30,40):
                    save_checkpoint(run_dir / f"epoch_{epoch:02d}.pt", model, arm, value, initial_box_hash, initial_q_hash, epoch, seed)
                print(json.dumps({"arm": arm, "epoch": epoch, "steps": total_steps,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
        if total_steps != 4150:
            raise RuntimeError(f"Pilot total steps {total_steps}, expected 4150")
        checkpoint = run_dir / "epoch_50.pt"
        save_checkpoint(checkpoint, model, arm, value, initial_box_hash, initial_q_hash, 50, seed)
        verify_freeze(model, frozen_initial)
        status.update({"status": "PASS", "checkpoint": str(checkpoint),
                       "checkpoint_sha256": sha(checkpoint),
                       "total_optimizer_steps": total_steps, "freeze_verified": True,
                       "elapsed_seconds": time.monotonic() - started})
        atomic_json(run_dir / "status.json", status)
    except BaseException as exc:
        status.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                       "total_optimizer_steps": total_steps})
        atomic_json(run_dir / "status.json", status)
        raise


