#!/usr/bin/env python3
"""Generate one illustration per Value & Growth offering, retaining source prompts.

Read business/value-growth.json and save images in business/media/. Existing
images are reused by default. --dry-run and --json-only require no API key.
Ordinary builds copy referenced images without invoking this script.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import time

TOOLKIT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TOOLKIT / '_wiring'))
import project_paths
from domain_paths import list_domain_files
from value_growth_model import offering_image_path, reference_catalog, validate_value_growth
from generate_customer_icons_gemini_nanobanana_api import (
    GeneratedImage, IMAGE_FORMATS, image_exists, write_bytes,
)
from generate_tutorial_images_gemini_nanobanana_api import DEFAULT_MODEL, call_gemini_nanobanana_api


@dataclass
class OfferingTarget:
    content: dict
    key: str
    image_path: Path
    prompt: str
    alt: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--domain', help='Domain ID or folder; omit to process all Value & Growth models.')
    scope.add_argument('--domain-dir', type=Path, help='Full domain folder; also selects the data project.')
    parser.add_argument('--project', type=Path, help='Data project root (default: current project).')
    parser.add_argument('--offering', help='One exact segment ID, shown by --dry-run.')
    parser.add_argument('--limit', type=int, default=0, help='Maximum new images across the selected scope; 0 means all.')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--aspect-ratio', choices=('16:9', '3:2', '4:3', '1:1'), default='16:9')
    parser.add_argument('--image-size', choices=('1K', '2K', '4K'), default='2K',
                        help='Default: 2K for readable value and pricing labels; use 4K for larger diagrams.')
    parser.add_argument('--dry-run', action='store_true', help='Preview actions without writes or API calls.')
    parser.add_argument('--show-prompts', action='store_true')
    parser.add_argument('--json-only', action='store_true', help='Link existing images without calling the API.')
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument('--skip-existing', action='store_true', help='Reuse existing images (the default).')
    existing.add_argument('--overwrite', action='store_true', help='Regenerate selected offering images.')
    parser.add_argument('--api-key-env', choices=('GEMINI_API_KEY', 'GOOGLE_API_KEY'), default='GEMINI_API_KEY')
    parser.add_argument('--max-retries', type=int, default=4)
    parser.add_argument('--retry-delay-seconds', type=float, default=5.0)
    parser.add_argument('--sleep-seconds', type=float, default=0.0)
    args = parser.parse_args(argv)
    for name in ('limit', 'max_retries', 'retry_delay_seconds', 'sleep_seconds'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    if args.json_only and args.overwrite:
        parser.error('--json-only cannot regenerate images with --overwrite')
    return args


def build_prompt(payload, segment, catalog, aspect_ratio='16:9'):
    customers = '; '.join(catalog['customers'][ref['customerId']]['name'] for ref in segment['customerRefs'])
    scene = segment.get('illustrationPrompt') or (
        'Use four illustrated panels titled "Value unit", "Pricing unit", "Charging model" and '
        '"Proof of value". Show what improves for the customer, what the supplier meters for billing, '
        'who pays and how the bill is composed, and how a buyer can test the benefit. '
        'Use concrete counters, labeled invoice components and an unfilled comparison worksheet '
        'grounded in the commercial fields below. Explain only supported charging terms and metrics.')
    return f'''Use case: infographic-diagram.
Create one detailed illustrated commercial-model explainer for one offering in a Value & Growth page.
Domain: {payload['domainId']}
Scope: {payload.get('scope', '')}
Offering: {segment['name']}
Summary: {segment['summary']}
Customers: {customers or segment['selling']['users']}
Users: {segment['selling']['users']}
Buyer: {segment['selling']['buyers']}
Who pays: {segment['selling']['payers']}
Customer outcome: {segment['outcome']}
Value unit: {segment['valueUnit']}
Pricing unit: {segment['pricingUnit']}
Charging model: {segment['pricingModel']}
Proof of value: {segment['valueProof']}
Evidence status: {segment.get('evidence', {}).get('kind', 'unspecified')}
Evidence limits: {segment.get('evidence', {}).get('note', '')}

Authored commercial example — follow its labels, comparisons and boundaries:
{scene}

Teaching structure:
- Make all four commercial concepts prominent in a spacious two-column, two-row layout:
  top-left "Value unit", top-right "Pricing unit", bottom-left "Charging model",
  bottom-right "Proof of value". These are related concepts, not chronological process steps.
- Value unit: depict the customer's useful outcome and how it is counted or measured.
  Pricing unit: depict the quantity or entitlement used to calculate the provider's bill.
  Keep the two explicit even when they use similar units. A fee is not the customer's value.
- Charging model: identify the payer and service provider, then show the supported fixed,
  variable, tiered or bundled bill components. Distinguish alternatives with "OR" and
  additive components with "+" only when justified by the source. Show the fee base clearly.
  Separate customer funds, seller proceeds, platform income and provider fees where relevant.
- Proof of value: draw the proposed comparison and evidence, not a claimed improvement.
  Use matching cohort, period and scope; name the metrics and relevant costs or tradeoffs.
  Use blank baseline/pilot cells or record checks. Do not fabricate results or causal attribution.
- Draw counters, meters, invoice components, bundle contents or comparison worksheets that
  explain these concepts visually. People establish the buyer/payer context but do not dominate.
  Do not turn this into a payment lifecycle, onboarding sequence or generic customer journey.
- Keep any arrows local to a panel and label their meaning. A connection between a metric
  and a bill is a conceptual relationship, not a transfer of the entire payment volume as fees.

Visual style and text:
- Landscape composition at {aspect_ratio}; four balanced panels with clear gutters.
- Match the domain's customer jobs and journey illustrations: expressive adult cartoon
  characters, clean navy outlines, teal and warm neutral accents. Use teal for customer value,
  blue for pricing units, amber for charging terms and green for evidence consistently.
- Use large headings, concise captions, simple symbolic calculations and readable labels.
  Prefer around 120–180 visible words. Avoid four prose-heavy cards or miniature dashboards.
- Print the storyboard's quoted labels exactly. Other short labels must be grounded in the
  commercial source fields. Do not print drawing instructions, citations or URLs.
- Use only authored symbolic fee or measurement formulas. Do not invent rates, prices, balances,
  result percentages, timelines, discounts, fee triggers, accounting conclusions or guarantees.
  Do not imply all alternative pricing models apply together or every provider charges each item.
- No decorative growth charts, brand logos, watermarks, photorealism or 3D rendering.
  Keep agreement-specific eligibility and responsibilities bounded by the supplied context.
- One complete image per offering, with all four concepts visible inside the frame.
'''.strip()


def prompt_path(image_path, business_dir):
    path = image_path.with_suffix('.prompt.txt')
    if not path.resolve().is_relative_to(business_dir.resolve()):
        raise ValueError('prompt path resolves outside the business folder')
    return path


def build_targets_for_file(path, payload, offering=None, aspect_ratio='16:9'):
    domain = path.parents[1]
    if not path.resolve().is_relative_to(domain.resolve()):
        raise ValueError(f'Value & Growth source resolves outside the selected domain: {path}')
    errors = validate_value_growth(payload, domain.name, domain, check_image_files=False)
    if errors:
        raise ValueError(f'{path}:\n' + '\n'.join(errors))
    catalog = reference_catalog(domain)
    targets, stems, used_paths = [], set(), set()
    for segment in payload.get('segments', []):
        key = segment['id']
        stem = 'offering-' + re.sub(r'[^a-z0-9]+', '-', key).strip('-')
        if stem in stems:
            raise ValueError(f'Offering IDs produce the same filename {stem!r} in {path}')
        stems.add(stem)
        candidates = [offering_image_path(path.parent, f'media/{stem}.{fmt}') for fmt in IMAGE_FORMATS]
        if 'image' in segment:
            authored = offering_image_path(path.parent, segment['image']['src'])
            candidates = [authored] + [authored.with_suffix(f'.{fmt}') for fmt in IMAGE_FORMATS] + candidates
        for candidate in candidates:
            offering_image_path(path.parent, candidate.relative_to(path.parent.resolve()).as_posix())
            prompt_path(candidate, path.parent)
            if candidate.resolve() in used_paths:
                raise ValueError(f'Offerings share an image path: {candidate}')
        used_paths.update(candidate.resolve() for candidate in candidates)
        if offering and key != offering:
            continue
        selected = next((p for p in candidates if image_exists(p)), candidates[0])
        targets.append(OfferingTarget(segment, key, selected,
                                      build_prompt(payload, segment, catalog, aspect_ratio),
                                      f"{segment['name']}: {segment['outcome']}"))
    return targets


def link_image(target, business_dir):
    image = dict(target.content.get('image', {}))
    image['src'] = target.image_path.relative_to(business_dir.resolve()).as_posix()
    image.setdefault('alt', target.alt)
    if target.content.get('image') == image:
        return False
    target.content['image'] = image
    return True


def run_generation(args, api_key):
    domain_id, domain_dir = project_paths.domain_selection(args.domain, args.domain_dir)
    project_paths.configure(project_paths.resolve_project_root(args.project, domain_dir))
    files = list_domain_files('business/value-growth.json', domain_dir.name if domain_dir else domain_id,
                              project_paths.DOMAINS_ROOT)
    # Check the selected scope before any paid API calls or writes.
    prepared = []
    for path in files:
        original = path.read_bytes()
        payload = json.loads(original)
        targets = build_targets_for_file(path, payload, args.offering, args.aspect_ratio)
        prepared.append((path, original, payload, targets))
    if args.offering and not any(targets for _, _, _, targets in prepared):
        raise ValueError(f'No offering matches {args.offering!r}. Use --dry-run to list offering IDs.')
    if not any(targets for _, _, _, targets in prepared):
        print('No Value & Growth offerings found. Nothing to illustrate.')
        return 0
    needs_api = any(args.overwrite or not image_exists(target.image_path)
                    for _, _, _, targets in prepared for target in targets)
    if needs_api and not (args.dry_run or args.json_only or api_key):
        raise ValueError(f'Set {args.api_key_env} locally, or use --dry-run or --json-only.')
    generated, updates = 0, 0
    for path, original, payload, targets in prepared:
        changed = False
        for target in targets:
            generate = (not args.json_only and (not args.limit or generated < args.limit)
                        and (args.overwrite or not image_exists(target.image_path)))
            action = 'Generate' if generate else ('Reuse' if image_exists(target.image_path) else 'Skip missing')
            print(f'{action}: {target.key}\n  {target.image_path}', flush=True)
            if args.show_prompts:
                print(target.prompt, flush=True)
            if generate:
                if not args.dry_run:
                    result = call_gemini_nanobanana_api(api_key, target, args)
                    target.image_path = target.image_path.with_suffix(f'.{result.output_format}')
                    offering_image_path(path.parent, target.image_path.relative_to(path.parent.resolve()).as_posix())
                    prompt_file = prompt_path(target.image_path, path.parent)
                    write_bytes(target.image_path, result.image_bytes)
                    write_bytes(prompt_file, (target.prompt + '\n').encode('utf-8'))
                generated += 1
            if image_exists(target.image_path) and not args.dry_run and link_image(target, path.parent):
                if path.read_bytes() != original:
                    raise RuntimeError(f'Value & Growth changed during generation; saved the image but kept the newer JSON: {path}. Rerun to link it.')
                original = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
                write_bytes(path, original)
                changed = True
            if generate and args.sleep_seconds and not args.dry_run:
                time.sleep(args.sleep_seconds)
        updates += int(changed)
    print(f'{"Would generate" if args.dry_run else "Generated"} {generated} offering illustration(s). Updated {updates} Value & Growth file(s).')
    return 0


def main():
    args = parse_args()
    try:
        return run_generation(args, os.environ.get(args.api_key_env, '').strip())
    except (OSError, RuntimeError, ValueError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
