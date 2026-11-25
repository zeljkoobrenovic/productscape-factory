"""Check offering scope, safe source updates and resumability without paid calls."""
import base64
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate_offering_images_gemini_nanobanana_api as illustrations

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


class OfferingImages(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='offering images ')
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name).resolve()
        self.addCleanup(illustrations.project_paths.configure, illustrations.project_paths.PROJECT_ROOT)
        self.source = self.project / '_config/product-domains/sample/alpha/business/value-growth.json'
        first = {
            'id': 'team-work', 'name': 'Team workflows', 'summary': 'Help teams coordinate tasks.',
            'customerRefs': [], 'productIds': [], 'competitorIds': [],
            'outcome': 'Complete shared work.', 'valueUnit': 'Completed workflow', 'pricingUnit': 'Team',
            'pricingModel': 'Monthly software subscription.', 'valueProof': 'Observe successful workflows.',
            'selling': {'motion': 'bottom-up', 'users': 'Team members', 'champions': 'Team lead',
                        'buyers': 'Team lead', 'payers': 'Company', 'channels': ['Self-service'], 'rationale': 'Teams can start independently.'},
            'positioning': {'strategy': 'differentiated', 'alternative': 'Manual coordination',
                            'rationale': 'Less coordination effort.', 'conditions': ['Teams value time saved.']},
            'activation': {'entry': 'Trial', 'firstValue': 'Finish a shared task.', 'repeatValue': 'Complete work weekly.',
                           'expansion': 'More teams', 'frictions': ['Setup']},
            'evidence': {'kind': 'illustrative', 'note': 'Test fixture.', 'sourceIds': []},
            'illustrationPrompt': 'Show two teammates finishing one task. Label only “Done”.'}
        second = deepcopy(first)
        second.update(id='solo-work', name='Solo workflows')
        self.payload = {'schemaVersion': '1.0', 'domainId': 'alpha', 'status': 'draft',
                        'title': 'Value & Growth', 'segments': [first, second]}
        self.write()
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def write(self):
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.source.write_text(json.dumps(self.payload))

    def read(self):
        return json.loads(self.source.read_text())

    def args(self, *extra):
        return illustrations.parse_args(['--project', str(self.project), *extra])

    def snapshot(self):
        return {str(p.relative_to(self.project)): p.read_bytes() for p in self.project.rglob('*') if p.is_file()}

    def image(self, name='offering-team-work.png'):
        path = self.source.parent / 'media' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG)
        return path

    def test_prompt_uses_customer_value_and_authored_scene(self):
        target = illustrations.build_targets_for_file(self.source, self.payload)[0]
        self.assertEqual(target.image_path.name, 'offering-team-work.png')
        for text in ('Complete shared work.', 'Monthly software subscription.', 'Team lead',
                     'Show two teammates', '16:9', 'navy outlines', 'not the customer'):
            self.assertIn(text, target.prompt)

    def test_default_explainer_keeps_commercial_fields_and_roles_distinct(self):
        person = {'name': 'Operations lead', 'jobsToBeDone': [
            {'id': 'release', 'name': 'Release reviewed work', 'outcome': 'Approved work delivered.',
             'steps': [{'step': 'Release the work', 'description': 'Send only approved items.'}]}]}
        segment = deepcopy(self.payload['segments'][0])
        del segment['illustrationPrompt']
        segment['customerRefs'] = [{'customerId': 'operator', 'jobIds': ['release'], 'kpiIds': []}]
        prompt = illustrations.build_prompt(self.payload, segment, {'customers': {'operator': person}})
        for expected in ('Value unit: Completed workflow', 'Pricing unit: Team',
                         'Charging model: Monthly software subscription.',
                         'Proof of value: Observe successful workflows.',
                         'Customers: Operations lead', 'Buyer: Team lead', 'Who pays: Company',
                         'Evidence status: illustrative', 'Evidence limits: Test fixture.'):
            self.assertIn(expected, prompt)
        self.assertNotIn('Send only approved items.', prompt)
        self.assertNotIn('Release the work', prompt)

    def test_dry_run_empty_scope_and_json_only_make_no_api_calls_or_writes(self):
        before = self.snapshot()
        with patch.object(illustrations, 'call_gemini_nanobanana_api') as api:
            illustrations.run_generation(self.args('--dry-run', '--show-prompts', '--limit', '1'), '')
            self.assertIn('Would generate 1', self.output.getvalue())
            illustrations.run_generation(self.args('--json-only'), '')
            self.assertEqual(before, self.snapshot())
            self.source.unlink()
            self.assertEqual(illustrations.run_generation(self.args(), ''), 0)
            api.assert_not_called()

    def test_missing_key_and_unknown_offering_fail_before_writes(self):
        before = self.snapshot()
        for args, error in ((self.args(), 'GEMINI_API_KEY'),
                            (self.args('--offering', 'absent', '--dry-run'), 'No offering matches')):
            with self.assertRaisesRegex(ValueError, error):
                illustrations.run_generation(args, '')
        self.assertEqual(before, self.snapshot())

    def test_single_offering_selection_and_limit_leave_other_source_content_intact(self):
        customer_file = self.source.parent.parent / 'customers/customers.json'
        customer_file.parent.mkdir()
        customer_file.write_text('[]\n')
        with patch.object(illustrations, 'call_gemini_nanobanana_api',
                          return_value=illustrations.GeneratedImage(PNG, 'png')) as api:
            illustrations.run_generation(self.args('--offering', 'solo-work', '--limit', '1'), 'fake-key')
            api.assert_called_once()
        updated = self.read()
        self.assertNotIn('image', updated['segments'][0])
        self.assertEqual(updated['segments'][1]['image']['src'], 'media/offering-solo-work.png')
        del updated['segments'][1]['image']
        self.assertEqual(updated, self.payload)
        self.assertEqual(customer_file.read_text(), '[]\n')

    def test_failure_checkpoints_and_resume_skips_successful_images(self):
        result = illustrations.GeneratedImage(PNG, 'png')
        with patch.object(illustrations, 'call_gemini_nanobanana_api', side_effect=[result, RuntimeError('busy')]):
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                illustrations.run_generation(self.args(), 'fake-key')
        self.assertIn('image', self.read()['segments'][0])
        self.assertNotIn('image', self.read()['segments'][1])
        self.assertIn('Show two teammates', (self.source.parent / 'media/offering-team-work.prompt.txt').read_text())
        with patch.object(illustrations, 'call_gemini_nanobanana_api', return_value=result) as api:
            illustrations.run_generation(self.args(), 'fake-key')
            api.assert_called_once()
            self.assertEqual(api.call_args.args[1].key, 'solo-work')
        before = self.snapshot()
        illustrations.run_generation(self.args('--skip-existing'), '')
        self.assertEqual(before, self.snapshot())

    def test_json_only_recovers_format_and_preserves_reviewed_alt(self):
        self.payload['segments'][0]['image'] = {'src': 'media/reviewed.png', 'alt': 'Reviewed alt text.'}
        self.write()
        self.image('reviewed.webp')
        with patch.object(illustrations, 'call_gemini_nanobanana_api') as api:
            illustrations.run_generation(self.args('--json-only'), '')
            api.assert_not_called()
        self.assertEqual(self.read()['segments'][0]['image'], {'src': 'media/reviewed.webp', 'alt': 'Reviewed alt text.'})
        self.assertNotIn('image', self.read()['segments'][1])
        before = self.snapshot()
        illustrations.run_generation(self.args('--json-only'), '')
        self.assertEqual(before, self.snapshot())

    def test_overwrite_preserves_alt_and_uses_returned_format(self):
        self.image()
        self.payload['segments'][0]['image'] = {'src': 'media/offering-team-work.png', 'alt': 'Reviewed.'}
        self.write()
        with patch.object(illustrations, 'call_gemini_nanobanana_api',
                          return_value=illustrations.GeneratedImage(b'mocked JPEG response', 'jpg')) as api:
            illustrations.run_generation(self.args('--overwrite', '--limit', '1'), 'fake-key')
            api.assert_called_once()
            self.assertEqual(api.call_args.args[2].image_size, '2K')
        self.assertEqual(self.read()['segments'][0]['image'], {'src': 'media/offering-team-work.jpg', 'alt': 'Reviewed.'})
        self.assertEqual((self.source.parent / 'media/offering-team-work.jpg').read_bytes(), b'mocked JPEG response')

    def test_concurrent_author_edits_are_preserved_and_saved_image_is_recoverable(self):
        def edited_during_call(*args):
            revised = self.read()
            revised['segments'][0]['summary'] = 'Author revised this during generation.'
            self.source.write_text(json.dumps(revised))
            return illustrations.GeneratedImage(PNG, 'png')
        with patch.object(illustrations, 'call_gemini_nanobanana_api', side_effect=edited_during_call):
            with self.assertRaisesRegex(RuntimeError, 'kept the newer JSON'):
                illustrations.run_generation(self.args('--limit', '1'), 'fake-key')
        self.assertNotIn('image', self.read()['segments'][0])
        illustrations.run_generation(self.args('--json-only'), '')
        self.assertIn('image', self.read()['segments'][0])
        self.assertEqual(self.read()['segments'][0]['summary'], 'Author revised this during generation.')

    def test_unsafe_paths_and_filename_collisions_fail_before_api(self):
        for src in ('../outside.png', '/tmp/outside.png', 'media/../../outside.png',
                    'https://example.org/image.png', 'media/bad.svg'):
            with self.subTest(src=src):
                self.payload['segments'][0]['image'] = {'src': src, 'alt': 'Example.'}
                self.write()
                with patch.object(illustrations, 'call_gemini_nanobanana_api') as api:
                    with self.assertRaises(ValueError):
                        illustrations.run_generation(self.args(), 'fake-key')
                    api.assert_not_called()
        del self.payload['segments'][0]['image']
        self.payload['segments'][1]['id'] = 'team:work'
        self.write()
        with self.assertRaisesRegex(ValueError, 'same filename'):
            illustrations.run_generation(self.args('--dry-run'), '')

    def test_image_and_prompt_symlinks_cannot_escape_business_folder(self):
        media = self.source.parent / 'media'
        media.mkdir()
        outside = self.project / 'outside'
        outside.write_bytes(b'original')
        for name in ('offering-team-work.png', 'offering-team-work.prompt.txt'):
            link = media / name
            link.symlink_to(outside)
            with patch.object(illustrations, 'call_gemini_nanobanana_api') as api:
                with self.assertRaisesRegex(ValueError, 'outside the business folder'):
                    illustrations.run_generation(self.args('--overwrite'), 'fake-key')
                api.assert_not_called()
            self.assertEqual(outside.read_bytes(), b'original')
            link.unlink()


if __name__ == '__main__':
    unittest.main()
