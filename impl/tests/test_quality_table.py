import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from core.figures import make_all, quality_table
from core.pairs import load_pair


FIXTURE_PATH = Path(__file__).resolve().parent / "e3_partial_b_only.json"


def _full_sample():
    kid = {"mean": 0.1, "std": 0.01, "subset_size": 2, "n_subsets": 2}
    return {
        "metrics_source": "real",
        "n_samples": 2,
        "mean": {"ssim_bc": 0.9, "psnr_bc": 29},
        "deep": {
            "lpips": {"n_images": 2, "L_ab": 0, "L_bc": 0.1, "L_ac": 0.1},
            "fid": {"n_real": 2, "b_vs_real_zebra": 1, "c_vs_real_zebra": 2},
            "kid": {"b_vs_real_zebra": kid, "c_vs_real_zebra": kid},
            "mask": {
                "mean_iou_ac": 0.6,
                "mean_dice_ac": 0.75,
                "detected_pairs": 1,
                "mean_background_ssim_ac": 0.9,
            },
            "canny_edge_ssim_bc": 0.7,
            "zebra_discrimination": {
                "b": {"top1_accuracy": 0.5, "n": 2},
                "c": {"top1_accuracy": 0.5, "n": 2},
            },
            "anomalies": {"count": 0},
        },
    }


class QualityTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h2z = load_pair("horse2zebra")
        cls.z2h = load_pair("zebra2horse")
        cls.winter = load_pair("summer2winter")
        cls.monet = load_pair("style_monet")
        cls.fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")

    def _export(self, payload, pair=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out_dir = root / "figures"
            (root / "quality.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            dest = quality_table(root, out_dir, pair=pair)
            self.assertTrue(dest.is_file())
            return dest.read_text(encoding="utf-8")

    def test_dice_is_parallel_with_iou_and_old_json_is_missing(self):
        q = _full_sample()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for value in (0.75, None):
                if value is None:
                    del q["deep"]["mask"]["mean_dice_ac"]
                (root / "quality.json").write_text(json.dumps(q), encoding="utf-8")
                text = quality_table(root, root, self.h2z).read_text(encoding="utf-8")
                self.assertIn("掩码 IoU(a,c)", text)
                expected = "0.7500" if value is not None else "缺失"
                denominator = "1" if value is not None else "未记录"
                self.assertIn(f"掩码 Dice(a,c) | {expected} | 有效 {denominator}/2 对", text)

    def test_single_batch_b_only_and_c_only_and_independent_denominators(self):
        # 1) 完整样本仅删 C 批与仅删 B 批
        q_b = _full_sample()
        q_b["metrics_source"] = "partial"
        del q_b["deep"]["zebra_discrimination"]
        q_b["deep"]["class_discrimination"] = {
            "target": "zebra",
            "by_batch": {"b": {"top1_accuracy": 0.75, "n": 12}},
        }
        text_b = self._export(q_b, self.h2z)
        self.assertIn("| 任务保真 | zebra 判别 top1（B / C） | 75.00% / 缺失 | B 批 n=12；缺少 C 批结果 |", text_b)

        q_c = copy.deepcopy(q_b)
        q_c["deep"]["class_discrimination"]["by_batch"] = {"c": {"top1_accuracy": 0.625}}
        text_c = self._export(q_c, self.h2z)
        self.assertIn(
            "| 任务保真 | zebra 判别 top1（B / C） | 缺失 / 62.50% | 缺少 B 批结果；C 批样本量未记录 |",
            text_c,
        )

        # 2) 两批完整（B=0, C=1 且分母不同，另测半分位 0.03125 / 0.09375）
        q_zero_one = _full_sample()
        del q_zero_one["deep"]["zebra_discrimination"]
        q_zero_one["deep"]["class_discrimination"] = {
            "target": "zebra",
            "by_batch": {
                "b": {"top1_accuracy": 0, "n": 11},
                "c": {"top1_accuracy": 1, "n": 19},
            },
        }
        text_zo = self._export(q_zero_one, self.h2z)
        self.assertIn("| 任务保真 | zebra 判别 top1（B / C） | 0.00% / 100.00% | B 批 n=11，C 批 n=19；差异应据实解释 |", text_zo)

        q_half_even = copy.deepcopy(q_zero_one)
        q_half_even["deep"]["class_discrimination"]["by_batch"] = {
            "b": {"top1_accuracy": 0.03125, "n": 16},
            "c": {"top1_accuracy": 0.09375, "n": 32},
        }
        text_he = self._export(q_half_even, self.h2z)
        self.assertIn("3.12% / 9.38%", text_he)

    def test_six_reading_groups_missing_and_shared_fixture_and_disabled_pairs(self):
        # E3-01 + E3-06：直接读取 e3_partial_b_only.json（缺 LPIPS/FID/KID/Canny/掩码/异常，SSIM/PSNR 为 null）
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out_dir = root / "figures"
            (root / "quality.json").write_text(self.fixture_text, encoding="utf-8")
            produced = make_all(root, out_dir, self.h2z)
            self.assertEqual(produced, [out_dir / "quality_table.md"])
            text = (out_dir / "quality_table.md").read_text(encoding="utf-8")

        self.assertIn("# E3 量化质量对比表（horse2zebra：输入总数 n=20 / zebra 参照数未记录）", text)
        self.assertIn("| 转换保真 | LPIPS(a,b) | 缺失 |", text)
        self.assertIn("| 逐像素 | SSIM(b,c) / PSNR(b,c) | 缺失 | 均值 |", text)
        self.assertIn("| 结构（边缘） | Canny 边缘 SSIM(b,c) | 缺失 |", text)
        self.assertIn("| 迁移质量 | FID(B vs 真实zebra) | 缺失 | FP32 批 |", text)
        self.assertIn("| 迁移质量 | KID(C) | 缺失 | INT8 批 |", text)
        self.assertIn("| 结构（前景/背景） | 掩码 IoU / Dice / 背景 SSIM | 缺失 | 缺失（未取得有效掩码结果） |", text)
        self.assertIn("| 任务保真 | zebra 判别 top1（B / C） | 75.00% / 缺失 | B 批 n=12；缺少 C 批结果 |", text)
        self.assertIn("| 异常登记 | SSIM(b,c)<0.80 | 未记录 | 见 fig_e3_anomalies.png |", text)

        # Canny=0.0 与异常数=0 均为有效零值，且单侧 B 批 FID/KID 可见而 C 侧标缺失
        q_partial_c = json.loads(self.fixture_text)
        q_partial_c["deep"]["canny_edge_ssim_bc"] = 0.0
        q_partial_c["deep"]["fid"] = {"n_real": 14, "b_vs_real_zebra": 81.25}
        q_partial_c["deep"]["kid"] = {
            "b_vs_real_zebra": {"mean": 0.0234, "std": 0.005, "subset_size": 10, "n_subsets": 20}
        }
        q_partial_c["deep"]["anomalies"] = {"count": 0}
        text_partial_c = self._export(q_partial_c, self.h2z)
        self.assertIn("| 结构（边缘） | Canny 边缘 SSIM(b,c) | 0.0000 |", text_partial_c)
        self.assertIn("| 迁移质量 | FID(B vs 真实zebra) | 81.25 | FP32 批 |", text_partial_c)
        self.assertIn("| 迁移质量 | FID(C vs 真实zebra) | 缺失 | INT8 批 |", text_partial_c)
        self.assertIn("| 迁移质量 | KID(B) | 0.0234 ± 0.0050（subset 10×20） | FP32 批 |", text_partial_c)
        self.assertIn("| 迁移质量 | KID(C) | 缺失 | INT8 批 |", text_partial_c)
        self.assertIn("| 异常登记 | SSIM(b,c)<0.80 | 0 例 |", text_partial_c)

        # E3-05：禁用对即使放入同一份含 75% 的 JSON，也只显示不适用原因
        for disabled_pair in (self.winter, self.monet):
            with self.subTest(pair=disabled_pair.id):
                text_dis = self._export(json.loads(self.fixture_text), disabled_pair)
                self.assertIn(disabled_pair.metric_reason("mask_iou"), text_dis)
                self.assertIn(
                    f"| 任务保真 | 类别判别 | 不适用 | 类别判别不适用：{disabled_pair.metric_reason('class_discrimination')} |",
                    text_dis,
                )
                self.assertNotIn("75.00%", text_dis)

        # E3-07：proxy 无 deep 时保持原导表拒绝提示
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "quality.json").write_text(json.dumps({"metrics_source": "proxy", "n_samples": 4}), encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                quality_table(root, root / "figures", self.h2z)
            self.assertIn("没有 deep 字段", str(ctx.exception))

    def test_legacy_reference_counts_dice_and_explicit_conflicts(self):
        q = json.loads((FIXTURE_PATH.parents[1] / "artifacts/results/quality.json").read_text("utf-8"))
        positive = self._export(q, self.h2z)
        self.assertIn("zebra 参照批 n=140", positive)
        self.assertIn("有效 未记录/120 对", positive)
        winter = self._export(q, self.winter)
        self.assertIn("winter 参照数未记录", winter)
        self.assertNotIn("80.70", winter)
        q["deep"]["fid"].update(ref_domain="winter", b_vs_real_zebra=0)
        q["deep"]["kid"]["ref_domain"] = "winter"
        for value, expected in ((None, "未记录"), (0.0, "110"), (0.7, "110")):
            q["deep"]["mask"]["mean_dice_ac"] = value
            text = self._export(q, self.h2z)
            self.assertIn("zebra 参照数未记录", text)
            self.assertIn("FID(B vs 真实zebra) | 缺失", text)
            self.assertIn("KID(B) | 缺失", text)
            dice_line = next(l for l in text.splitlines() if "掩码 Dice(a,c)" in l)
            self.assertIn(f"有效 {expected}/120 对", dice_line)
        q["deep"]["fid"] = {"n_real": 140}
        self.assertIn("zebra 参照数未记录", self._export(q, self.h2z))

    def test_structure_side_table_is_independent_and_preserves_zero_valid(self):
        from contextlib import redirect_stdout
        from io import StringIO
        q = json.loads(self.fixture_text)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "figures"
            main, side = out / "quality_table.md", out / "structure_table_v2.md"
            for pairs, expected in (
                    ({"a_c": {"iou": {"mean": 0.5}}}, "0.500000 | 未记录/未记录"),
                    ({"a_c": {"iou": {"mean": None, "n_valid": 0, "n_total": 12}}}, "缺失 | 0/12"),
                    ({}, None)):
                q["deep"]["structure"] = {"pairs": pairs}
                (root / "quality.json").write_text(json.dumps(q), encoding="utf-8")
                log = StringIO()
                with redirect_stdout(log):
                    paths = make_all(root, out, self.h2z)
                self.assertIn("75.00% / 缺失", main.read_text(encoding="utf-8"))
                self.assertNotIn("[跳过] E3 质量对比表", log.getvalue())
                if expected:
                    self.assertEqual(paths, [main, side])
                    self.assertIn(expected, side.read_text(encoding="utf-8"))
                else:
                    self.assertEqual(paths, [main])
                    self.assertIn("[跳过] E3 结构评测 v2 副表", log.getvalue())
                    self.assertIn("本次未生成", log.getvalue())

    @unittest.skipUnless(os.name == "nt" and (FIXTURE_PATH.parents[1] / "run-local.ps1").is_file(),
                         "requires the Windows launcher in a source checkout")
    def test_launcher_figures_keeps_main_table_for_empty_structure(self):
        q = json.loads(self.fixture_text)
        q["deep"]["structure"] = {"pairs": {}}
        with tempfile.TemporaryDirectory(prefix="ShiftLite E3 交接-") as tmp:
            art = Path(tmp) / "impl/artifacts"
            (art / "results").mkdir(parents=True)
            (art / "results/quality.json").write_text(json.dumps(q), encoding="utf-8")
            env = dict(os.environ, E3_RUN=tmp, E3_PYTHON=sys.executable,
                       E3_LAUNCHER=str(FIXTURE_PATH.parents[1] / "run-local.ps1"))
            command = ("[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); "
                       "& $env:E3_LAUNCHER -Python $env:E3_PYTHON -RunRoot $env:E3_RUN "
                       "-CliArgs @('figures','--pair','horse2zebra'); exit $LASTEXITCODE")
            out = subprocess.check_output(["powershell.exe", "-NoProfile", "-ExecutionPolicy",
                                          "RemoteSigned", "-Command", command], env=env,
                                         stderr=subprocess.STDOUT, timeout=90).decode("utf-8")
            self.assertIn("[跳过] E3 结构评测 v2 副表", out)
            self.assertNotIn("[跳过] E3 质量对比表", out)
            self.assertIn("quality_table.md", out)
            self.assertFalse((art / "figures/structure_table_v2.md").exists())
            self.assertIn("75.00% / 缺失", (art / "figures/quality_table.md").read_text("utf-8"))

    def test_rounded_classification_matches_web_contract(self):
        q = _full_sample()
        for batch, value in zip(q["deep"]["zebra_discrimination"].values(), [0.38835, 0.61165]):
            batch["top1_accuracy"] = value
        self.assertIn("38.83% / 61.17%", self._export(q))


if __name__ == "__main__":
    unittest.main()
