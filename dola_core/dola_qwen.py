import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class DoLaQwen:
    def __init__(self, model_name, device="cuda", num_gpus=1, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = int(num_gpus)
        self.max_gpu_memory = max_gpu_memory
        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        import subprocess, os
        # Detect available GPUs by checking free memory
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            free, total = line.strip().split(', ')
            gpu_info.append((int(free), int(total)))

        # Respect CUDA_VISIBLE_DEVICES if set
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible:
            visible = [int(x.strip()) for x in cuda_visible.split(',')]
            gpu_info = [(gpu_info[i][0], gpu_info[i][1]) for i in visible]
            available_gpus = list(range(len(visible)))
        else:
            # Pick GPUs with enough free memory (need ~20GB for 8B fp16)
            available_gpus = [i for i, (free, _) in enumerate(gpu_info) if free > 20000]
            if not available_gpus:
                # Fallback: use all GPUs
                available_gpus = list(range(len(gpu_info)))

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if len(available_gpus) == 1:
            gpu_id = available_gpus[0]
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=True, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map={"": gpu_id})
            self.device_map = False
            self._device = f"cuda:{gpu_id}"
        else:
            max_mem = {i: f"{min(gpu_info[i][0] - 2000, self.max_gpu_memory * 1024)}MiB"
                       for i in available_gpus}
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=True, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto", max_memory=max_mem)
            self.device_map = True
            self._device = "cuda:0"

        model.eval()
        return model, tokenizer

    def get_logits_at_layers(self, input_ids, early_exit_layers):
        """Forward pass returning logits at specified layers via output_hidden_states."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # tuple of (num_layers+1,) each (1, seq_len, hidden)

            logits_dict = {}
            for layer_idx in early_exit_layers:
                # hidden_states[0] is embedding output, hidden_states[i] is output of layer i
                hs = hidden_states[layer_idx]
                logits = self.model.lm_head(hs)
                logits_dict[layer_idx] = logits
        return logits_dict

    def get_relative_top_filter(self, scores, relative_top=0.1, min_tokens_to_keep=1):
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def _contrast_logits(self, mature_logits, base_logits, relative_top=0.0,
                         relative_top_value=-1000.0, post_softmax=True):
        """Compute DoLa contrastive logits: mature - base, with optional post-softmax and relative top filtering."""
        mature_log_softmax = mature_logits.log_softmax(dim=-1)
        base_log_softmax = base_logits.log_softmax(dim=-1)
        diff_logits = mature_log_softmax - base_log_softmax
        if post_softmax:
            diff_logits = diff_logits.log_softmax(dim=-1)
        if relative_top > 0.0:
            mask = self.get_relative_top_filter(mature_log_softmax, relative_top)
            diff_logits = torch.where(mask, relative_top_value, diff_logits)
        return diff_logits

    def _pick_premature_layer(self, logits_dict, mature_layer, candidate_premature_layers, seq_i):
        """Pick the premature layer with maximum JS divergence from the mature layer at position seq_i."""
        stacked_premature = torch.stack(
            [logits_dict[l][:, seq_i, :] for l in candidate_premature_layers], dim=0
        )
        softmax_mature = F.softmax(logits_dict[mature_layer][:, seq_i, :], dim=-1)
        softmax_premature = F.softmax(stacked_premature, dim=-1)

        M = 0.5 * (softmax_mature[None, :, :] + softmax_premature)
        log_softmax_mature = F.log_softmax(logits_dict[mature_layer][:, seq_i, :], dim=-1)
        log_softmax_premature = F.log_softmax(stacked_premature, dim=-1)

        kl1 = F.kl_div(log_softmax_mature[None, :, :], M, reduction='none').mean(-1)
        kl2 = F.kl_div(log_softmax_premature, M, reduction='none').mean(-1)
        js_divs = 0.5 * (kl1 + kl2).mean(-1)

        return candidate_premature_layers[int(js_divs.argmax().cpu().item())]

    def generate(self, input_text, max_new_tokens=50, top_p=0.95, top_k=0, temperature=0.9,
                 mode='baseline', mature_layer=None, premature_layer=None,
                 candidate_premature_layers=None, repetition_penalty=1.0,
                 relative_top=0.1, relative_top_value=-1000.0,
                 stop_strs=None, use_chat_template=False, **kwargs):
        """Open-ended generation with DoLa decoding."""
        with torch.no_grad():
            if use_chat_template and isinstance(input_text, list):
                input_ids = self.tokenizer.apply_chat_template(
                    input_text, return_tensors="pt", add_generation_prompt=True,
                    enable_thinking=False
                ).to(self._device)
            else:
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self._device)

            if stop_strs is None:
                stop_strs = []
            stop_ids = []
            for s in stop_strs:
                ids = self.tokenizer.encode('\n' + s, add_special_tokens=False)
                stop_ids.append(ids)

            premature_layer_dist = None
            if mode == 'dola' and candidate_premature_layers is not None:
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}

            generated = input_ids[0].tolist()
            past_key_values = None

            for _ in range(max_new_tokens):
                current_ids = torch.tensor([generated], dtype=torch.long, device=self._device)

                if mode == 'baseline':
                    outputs = self.model(input_ids=current_ids, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits[0, -1, :]
                    past_key_values = outputs.past_key_values

                elif mode in ('dola-static', 'dola'):
                    if past_key_values is None:
                        # Full forward on first step
                        all_layers = []
                        if mode == 'dola-static':
                            all_layers = [premature_layer, mature_layer]
                        else:
                            all_layers = candidate_premature_layers + [mature_layer]

                        out = self.model(input_ids=current_ids, output_hidden_states=True, use_cache=True, return_dict=True)
                        past_key_values = out.past_key_values
                        hidden_states = out.hidden_states

                        logits_dict = {}
                        for l in all_layers:
                            hs = hidden_states[l]
                            logits_dict[l] = self.model.lm_head(hs)
                    else:
                        # Incremental forward for subsequent steps
                        all_layers = []
                        if mode == 'dola-static':
                            all_layers = [premature_layer, mature_layer]
                        else:
                            all_layers = candidate_premature_layers + [mature_layer]

                        # Get hidden states of just the new token via the full model
                        out = self.model(input_ids=current_ids[:, -1:], past_key_values=past_key_values,
                                         output_hidden_states=True, use_cache=True, return_dict=True)
                        past_key_values = out.past_key_values
                        hidden_states = out.hidden_states

                        logits_dict = {}
                        for l in all_layers:
                            hs = hidden_states[l]
                            logits_dict[l] = self.model.lm_head(hs)

                    if mode == 'dola-static':
                        base_logits = logits_dict[premature_layer][0, -1, :]
                        final_logits = logits_dict[mature_layer][0, -1, :]
                        logits = self._contrast_logits(
                            final_logits, base_logits, relative_top, relative_top_value)
                    else:
                        picked = self._pick_premature_layer(
                            logits_dict, mature_layer, candidate_premature_layers, -1)
                        premature_layer_dist[picked] += 1
                        base_logits = logits_dict[picked][0, -1, :]
                        final_logits = logits_dict[mature_layer][0, -1, :]
                        logits = self._contrast_logits(
                            final_logits, base_logits, relative_top, relative_top_value)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated):
                        logits[token_id] /= repetition_penalty

                # Temperature
                if temperature > 0 and temperature != 1.0:
                    logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                # Check EOS
                if next_token == self.tokenizer.eos_token_id:
                    output_str = self.tokenizer.decode(generated[input_ids.shape[-1]:], skip_special_tokens=True)
                    for s in stop_strs:
                        if output_str.endswith(s):
                            output_str = output_str[:-len(s)].strip()
                    return output_str, premature_layer_dist

                # Check stop words
                if stop_ids:
                    for stop_seq in stop_ids:
                        if len(generated) >= len(stop_seq) and generated[-len(stop_seq):] == stop_seq:
                            output_str = self.tokenizer.decode(generated[input_ids.shape[-1]:], skip_special_tokens=True)
                            # Remove stop word from output
                            for s in stop_strs:
                                if output_str.endswith(s):
                                    output_str = output_str[:-len(s)].strip()
                            return output_str, premature_layer_dist

            output_str = self.tokenizer.decode(generated[input_ids.shape[-1]:], skip_special_tokens=True)
            # Remove stop word from output
            for s in stop_strs:
                if output_str.endswith(s):
                    output_str = output_str[:-len(s)].strip()
            return output_str, premature_layer_dist

    def lm_score(self, input_text1, input_text2, max_new_tokens=256, mode='baseline',
                 mature_layer=None, premature_layer=None, candidate_premature_layers=None,
                 relative_top=0.0, relative_top_value=-1000.0, post_softmax=True, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self._device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self._device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]

            if mode == 'baseline':
                outputs = self.model(input_ids).logits.squeeze(0)
                continue_ids = continue_ids.to(outputs.device)
                outputs = outputs.log_softmax(-1)
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                return log_probs, None

            elif mode == 'dola-static':
                assert premature_layer is not None and mature_layer is not None
                logits_dict = self.get_logits_at_layers(input_ids, [premature_layer, mature_layer])
                continue_ids = continue_ids.to(logits_dict[mature_layer].device)

                base_logits = logits_dict[premature_layer][:, prefix_ids.shape[-1] - 1: -1, :].squeeze(0)
                final_logits = logits_dict[mature_layer][:, prefix_ids.shape[-1] - 1: -1, :].squeeze(0)

                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                return log_probs, None

            elif mode == 'dola':
                assert mature_layer is not None and candidate_premature_layers is not None
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}

                all_layers = candidate_premature_layers + [mature_layer]
                logits_dict = self.get_logits_at_layers(input_ids, all_layers)
                continue_ids = continue_ids.to(logits_dict[mature_layer].device)

                picked_layers = []
                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Keep batch dim: shape (num_premature, batch=1, vocab)
                    stacked_premature = torch.stack(
                        [logits_dict[l][:, seq_i, :] for l in candidate_premature_layers], dim=0
                    )
                    # shape: (batch=1, vocab)
                    softmax_mature = F.softmax(logits_dict[mature_layer][:, seq_i, :], dim=-1)
                    # shape: (num_premature, batch=1, vocab)
                    softmax_premature = F.softmax(stacked_premature, dim=-1)

                    # M: (num_premature, batch=1, vocab)
                    M = 0.5 * (softmax_mature[None, :, :] + softmax_premature)
                    log_softmax_mature = F.log_softmax(logits_dict[mature_layer][:, seq_i, :], dim=-1)
                    log_softmax_premature = F.log_softmax(stacked_premature, dim=-1)

                    kl1 = F.kl_div(log_softmax_mature[None, :, :], M, reduction='none').mean(-1)
                    kl2 = F.kl_div(log_softmax_premature, M, reduction='none').mean(-1)
                    js_divs = 0.5 * (kl1 + kl2)
                    js_divs = js_divs.mean(-1)  # (num_premature,)

                    picked = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[picked] += 1
                    picked_layers.append(picked)

                # shape: (seq_len, vocab)
                base_logits = torch.zeros_like(logits_dict[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(picked_layers):
                    base_logits[i] = logits_dict[l][0, prefix_ids.shape[-1] - 1 + i]

                final_logits = logits_dict[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                return log_probs, premature_layer_dist
