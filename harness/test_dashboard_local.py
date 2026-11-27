"""新首页的回环浏览器验收，所有席位、计时和进程状态均为隔离夹具。"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from threading import Thread

from playwright.sync_api import expect, sync_playwright

from harness import dashboard
from harness.dashboard_local import campaign_view, file_for, latency_trend, top_sources
from harness.test_local_eval import complete


class LocalDashboardTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())) / 'campaign'
        self.root.mkdir()

    def seed(self):
        source = self.root / 'solution.cu'
        source.write_bytes(b'// fixture only\n')
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        report = complete()
        report['source_sha256'] = sha
        raw = self.root / 'r01.json'
        raw.write_text(json.dumps(report))
        shape_ids = [s['shape_id'] for s in report['shapes']]
        shapes = dict.fromkeys(shape_ids, 5.5)
        item = dict(id='r01', rankable=True, status='ok', source_sha256=sha, source=str(source), report=str(raw),
                    agent='c', round_id='c-r1', total_us=44, shape_means_us=shapes)
        seats = {a: dict(agent=a, round=1, round_id=a+'-r1', parent_sha=sha, candidate_sha=sha, source=str(source),
                         state=state, detail=detail, last_activity='2025-01-01T09:00:00+00:00') for a,state,detail in (
                         ('a','developing','准备新机制'),('b','queued','等待单卡释放'),('c','evaluating','第四形状'))}
        state = dict(problem='leaky_relu', agents=seats, shape_ids=shape_ids, rounds={}, reports=[item], best=item,
                     device=dict(state='busy',reason='c 的 ABBA'),
                     groups={'g':dict(status='queued')}, shape_routes=[dict(shape_id=shape_ids[0], source_sha256=sha,
                     control_sha='e'*64,gains_us=[2,3],band_us=1)])
        (self.root/'state.json').write_text(json.dumps(state))
        return state, sha

    def test_preparation_is_read_only_and_files_must_be_indexed_inside_campaign(self):
        (self.root/'preparation.json').write_text(json.dumps(dict(stage='架构验收', detail='设备待接入')))
        view = campaign_view(self.root)
        self.assertEqual(view['preparation']['stage'], '架构验收')
        self.assertEqual([s['agent'] for s in view['agents']], list('abc'))
        self.assertFalse((self.root/'state.json').exists())
        state, sha = self.seed()
        self.assertEqual(file_for(self.root,'source',sha),self.root/'solution.cu')
        self.assertIsNone(file_for(self.root,'report','../outside.json'))
        forbidden = self.root.parent/'outside.json'
        forbidden.write_text('private')
        state['reports'][0]['report'] = str(forbidden)
        (self.root/'state.json').write_text(json.dumps(state))
        self.assertIsNone(file_for(self.root,'report','r01'))

    def test_trend_orders_decisions_and_uses_only_candidate_legs(self):
        def leg(key, sha, side, t, ok=True):
            return dict(id=key, source_sha256=sha, side=side, rankable=ok, total_us=t,
                        recorded_at='10:00', shape_means_us={'small':t/2,'large':t/2})

        rows=[leg('cal1','base','A',90),leg('cal2','base','B',110),leg('control','base','A',200),
              leg('b1','new','B',51),leg('b2','new','B',55),leg('bad','new','B',1,False),
              leg('other','other','B',2)]
        round1 = dict(id='a-r0001', agent='a', round=1, decision='promoted', evaluated_at='10:02',
                      candidate_sha='new', report_ids=['control','b1','b2','bad','other'])
        round2 = dict(round1, id='b-r0001', agent='b', decision='regressed', evaluated_at='10:04',
                      candidate_sha='other', report_ids=['other'])
        state = dict(groups=[dict(kind='calibrate',status='completed',report_ids=['cal1','cal2'])],
                     shapes=[], shape_ids=['small','large'], reports=rows, rounds=[round2,round1,
                     dict(round1,decision='invalid'),dict(round1,decision='duplicate'),dict(round1,evaluated_at=None)])
        points=latency_trend(state)
        self.assertEqual([p['id'] for p in points], ['baseline','a-r0001','b-r0001'])
        self.assertEqual(points[1]['total_range_us'], [51,55])
        self.assertEqual([(p['samples'],p['total_us'],p['shape_means_us']['small']) for p in points],
                         [(2,100,50),(2,53,26.5),(1,2,1)])

    def test_loopback_browser_seats_all_shapes_samples_and_source(self):
        state, sha = self.seed()
        r = state['reports'][0]
        r.update(side='B',recorded_at='2025-01-01T09:00:00Z')
        state['groups']['g'] = dict(kind='calibrate',status='completed',report_ids=[r['id']])
        for n in (1, 2, 3):
            for agent in 'abc':
                key = f'{agent}-r{n:04d}'
                state['rounds'][key] = dict(id=key,agent=agent,round=n,decision='promoted' if agent=='a' else 'shared_best',
                    evaluated_at=f'2025-01-01T09:0{n}:00Z',candidate_sha=sha,report_ids=[r['id']],conclusion='已测')
        (self.root/'state.json').write_text(json.dumps(state))
        server = self.enterContext(dashboard.serve(0, [self.root]))
        thread = Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        origin = f'http://127.0.0.1:{server.server_port}'
        playwright = self.enterContext(sync_playwright())
        cache = Path(os.getenv('PLAYWRIGHT_BROWSERS_PATH', Path.home()/'.cache/ms-playwright'))
        paths = [Path(os.getenv('KEW_CHROMIUM') or playwright.chromium.executable_path),
                 *sorted(cache.glob('chromium-*/chrome-linux*/chrome'), reverse=True)]
        executable = next((p for p in paths if p.is_file() and os.access(p, os.X_OK)), None)
        if executable is None:
            self.skipTest('未找到用于回环夹具的 Chromium')
        browser = playwright.chromium.launch(executable_path=executable)
        self.addCleanup(browser.close)
        page = browser.new_page(viewport={'width':1200,'height':900})
        errors=[]
        requests=[]
        page.on('pageerror',lambda error: errors.append(str(error)))
        page.on('request',lambda req:requests.append(req.url))
        page.goto(origin)
        page.locator('#seats .box').first.wait_for()
        self.assertEqual(page.locator('#seats h3').all_text_contents(), ['Agent a','Agent b','Agent c'])
        self.assertEqual(page.locator('#seats .state').all_text_contents(), ['开发中','等待 H20','评测中'])
        self.assertIn('整题最佳', page.locator('#comparison').inner_text())
        self.assertEqual(page.locator('#comparison tr').count(), 11)
        self.assertTrue(page.locator('#trend-plot').is_visible())
        self.assertIn('44.000', page.locator('#trend-inspect').inner_text())
        self.assertEqual(page.locator('#trend-round option').all_text_contents(), ['第一轮','第二轮','第三轮'])
        self.assertEqual(page.locator('#trend-inspect tbody tr, #trend-inspect table > tr').count(), 4)
        self.assertEqual(page.locator('.shape-group').count(), 2)
        self.assertEqual(page.locator('.shape-plot').count(), 8)
        self.assertIn('c 5.50', page.locator('.shape-values').first.inner_text())
        page.locator('#trend-round').select_option('2')
        self.assertEqual(page.locator('#trend-current').inner_text(), '第二轮对照')
        page.locator('#trend-plot').press('Home')
        self.assertEqual(page.locator('#trend-current').inner_text(), '第一轮对照')
        self.assertIn('比基线', page.locator('#trend-inspect').inner_text())
        page.locator('#trend-baseline').check()
        self.assertEqual(page.locator('#trend-round option').count(), 3)
        self.assertIn('Agent a', page.locator('#trend-inspect').inner_text())
        self.assertTrue(page.get_by_role('button',name='查看 TOP 3 源码').is_disabled())
        page.locator('#results > details > summary').first.click()
        page.locator('#results .report-body > details > summary').first.click()
        self.assertIn('耗时不计分', page.locator('#results .report-body').first.inner_text())
        self.assertEqual(page.locator('#results .report-body > details').count(), 8)
        shape = page.locator('#results .report-body > details').first
        self.assertEqual(shape.locator('table tr').count(), 11)
        shape.locator('tbody details summary, td details summary').first.click()
        self.assertIn('GPU', shape.locator('td details').first.inner_text())
        self.assertEqual(page.get_by_role('link',name='打开原始 JSON').get_attribute('href'), '/api/campaign/0/report/r01')
        page.set_viewport_size({'width':390,'height':844})
        self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'),390)
        state['round_target'] = 3
        for seat in state['agents'].values():
            seat.update(round=3, state='idle')
        for row in state['rounds'].values():
            row.update(review={'conclusion':'完成'}, source=str(self.root/'solution.cu'))
        for agent, total in [('a', 50), ('b', 40)]:
            path = self.root / (agent+'.cu')
            path.write_text('// ' + agent + '\n// 可复制\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row = state['rounds'][agent+'-r0001']
            row.update(source=str(path), candidate_sha=digest, report_ids=[agent])
            state['reports'].append(dict(r,id=agent,round_id=row['id'],source=str(path),source_sha256=digest,total_us=total))
        (self.root/'state.json').write_text(json.dumps(state))
        page.get_by_role('button',name='查看 TOP 3 源码').click(timeout=8000)
        choices = page.get_by_role('radio')
        self.assertEqual(choices.count(), 3)
        self.assertEqual(page.locator('#rank-options strong').all_text_contents(),
                         ['1.　40.000 µs','2.　44.000 µs','3.　50.000 µs'])
        for idx, file in [(2,'a.cu'), (1,'solution.cu'), (0,'b.cu')]:
            choices.nth(idx).check()
            source = (self.root/file).read_text()
            expect(page.locator('#rank-source')).to_have_value(source)
            page.context.grant_permissions(['clipboard-read','clipboard-write'], origin=origin)
            page.locator('#rank-copy').click()
            expect(page.locator('#rank-status')).to_have_text('已复制完整源码')
            self.assertEqual(page.evaluate('navigator.clipboard.readText()'),source)
        self.assertLessEqual(page.evaluate('document.querySelector("#rank-dialog").scrollWidth'),390)
        # 自动刷新应保留用户在源码框中的选区。
        page.locator('#rank-source').focus()
        page.keyboard.press('Control+a')
        page.wait_for_event('response', predicate=lambda response: response.url.endswith('/api/local'),timeout=7000)
        self.assertEqual(page.locator('#rank-source').evaluate('e=>e.selectionEnd-e.selectionStart'),len(source))
        page.evaluate("""() => {
            Object.defineProperty(navigator,'clipboard',{value:undefined});
            Object.defineProperty(document,'execCommand',{value:()=>false});
        }""")
        page.locator('#rank-copy').click()
        expect(page.locator('#rank-status')).to_contain_text('已选中完整源码')
        self.assertEqual(page.locator('#rank-source').evaluate('e=>e.selectionEnd-e.selectionStart'),len(source))
        page.keyboard.press('Escape')
        self.assertFalse(page.get_by_role('dialog').is_visible())
        (self.root/'b.cu').write_text('// archived bytes changed')
        self.assertEqual(page.request.get(origin+'/api/campaign/0/source/'+state['rounds']['b-r0001']['candidate_sha']).status,409)
        self.assertFalse(errors)
        self.assertTrue(all(url.startswith(origin+'/') for url in requests))

    def test_finished_ranking_pools_distinct_candidate_legs_and_closes_on_extension(self):
        state, sha = self.seed()
        state['round_target'] = 1
        state['groups'] = {}
        for agent, total in [('a',10),('b',20),('c',100)]:
            state['agents'][agent].update(state='idle')
            leg = dict(state['reports'][0],id=agent,side='B',total_us=total)
            if agent == 'b':
                leg.update(source_sha256='b'*64)
            state['reports'].append(leg)
            state['rounds'][agent] = dict(id=agent,agent=agent,round=1,evaluated_at='now',decision='shape_only',
                review={'conclusion':'done'},candidate_sha=leg['source_sha256'],source=leg['source'],report_ids=[agent])
        state['rounds']['c']['report_ids'].extend(['a','bad','control'])
        state['reports'] += [dict(leg,id='bad',total_us=0,rankable=False),dict(leg,id='control',side='A',total_us=1)]
        state.update(agents=list(state['agents'].values()),rounds=list(state['rounds'].values()),queue=[])
        self.assertEqual([(r['source_sha256'],r['total_us'],r['samples']) for r in top_sources(state)],
                         [('b'*64,20,1),(sha,55,2)])
        for change in (dict(round_target=None),dict(round_target=2),dict(queue=[{'status':'running'}])):
            self.assertIsNone(top_sources(dict(state,**change)))
        state['rounds'][0]['review'] = None
        self.assertIsNone(top_sources(state))
