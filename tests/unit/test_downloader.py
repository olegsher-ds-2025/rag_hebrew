import ssl

import pytest

import config
import downloader.manager as manager_mod
from downloader.mavat_downloader import MavatDownloader, _LegacySSLAdapter

# ---------------------------------------------------------------- filenames

@pytest.mark.parametrize('raw,forbidden', [
    ('../../etc/passwd', '/'),
    ('a"b.pdf', '"'),
    ('<script>.pdf', '<'),
    ('a&b=c.pdf', '&'),
    ('x%00y.pdf', '%'),
    ("a'b.pdf", "'"),
])
def test_sanitize_filename_strips_dangerous_chars(raw, forbidden):
    out = MavatDownloader._sanitize_filename(raw)
    assert forbidden not in out


def test_sanitize_filename_strips_path_components():
    assert MavatDownloader._sanitize_filename('../../etc/passwd') == 'passwd'
    assert MavatDownloader._sanitize_filename(r'..\..\win\path.pdf') == 'path.pdf'


def test_sanitize_filename_keeps_hebrew_names():
    name = 'תכנית מס. 504-0100552 (הוראות).pdf'
    assert MavatDownloader._sanitize_filename(name) == name


def test_sanitize_filename_length_cap_and_empty():
    assert len(MavatDownloader._sanitize_filename('x' * 500 + '.pdf')) <= 150
    assert MavatDownloader._sanitize_filename('...') == 'doc'


# ---------------------------------------------------------------- TLS policy

def test_tls_verification_on_by_default(monkeypatch):
    monkeypatch.setattr(config.settings, 'mavat_insecure_ssl', False)
    ctx = _LegacySSLAdapter().poolmanager.connection_pool_kw['ssl_context']
    assert ctx.verify_mode != ssl.CERT_NONE
    assert ctx.check_hostname


def test_tls_escape_hatch_requires_flag(monkeypatch):
    monkeypatch.setattr(config.settings, 'mavat_insecure_ssl', True)
    ctx = _LegacySSLAdapter().poolmanager.connection_pool_kw['ssl_context']
    assert ctx.verify_mode == ssl.CERT_NONE


def test_legacy_adapter_scoped_to_gov_hosts():
    d = MavatDownloader()
    assert isinstance(d.session.get_adapter('https://mavat.iplan.gov.il/x'), _LegacySSLAdapter)
    assert isinstance(d.session.get_adapter('https://ags.iplan.gov.il/x'), _LegacySSLAdapter)
    assert not isinstance(d.session.get_adapter('https://example.com/x'), _LegacySSLAdapter)


# ---------------------------------------------------------------- ArcGIS query

def test_search_plans_sanitizes_where_clause(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'features': [{'attributes': {'pl_number': '504-1', 'pl_name': 'נוף', 'pl_url': 'u'}}]}

    d = MavatDownloader()

    def fake_get(url, params=None, timeout=None, **kw):
        captured['where'] = params['where']
        return FakeResp()

    monkeypatch.setattr(d.session, 'get', fake_get)
    plans = d._search_plans("נוף%'; DROP TABLE--")
    assert plans[0]['pl_number'] == '504-1'
    # The allowlist must strip LIKE wildcards and quotes from the user value;
    # the only % left are the fixed '%...%' wrapping of the two LIKE patterns.
    where = captured['where']
    assert where.count('%') == 4
    assert "';" not in where


# ---------------------------------------------------------------- download dedup

def test_existing_file_skipped_without_network(tmp_path, monkeypatch):
    d = MavatDownloader()
    d.log = []
    (tmp_path / 'doc.pdf').write_bytes(b'%PDF')

    def no_network(*a, **k):
        raise AssertionError('network must not be touched for existing files')

    monkeypatch.setattr(d.session, 'get', no_network)
    dest = d._download_pdf('504', '1', 'doc.pdf', tmp_path)
    assert dest == tmp_path / 'doc.pdf'


def test_download_url_is_encoded(tmp_path, monkeypatch):
    d = MavatDownloader()
    d.log = []
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=None):
            return iter([b'%PDF'])

    def fake_get(url, timeout=None, stream=None, **kw):
        captured['url'] = url
        return FakeResp()

    monkeypatch.setattr(d.session, 'get', fake_get)
    d._download_pdf('504', '9', 'שם עם רווח.pdf', tmp_path)
    assert ' ' not in captured['url']
    assert 'fn=' in captured['url']


# ---------------------------------------------------------------- metadata

def test_build_metadata_chunk_fields():
    d = MavatDownloader()
    meta = d._build_metadata_chunk({
        'pl_number': '504-0100552',
        'pl_name': 'נוף הפארק',
        'internet_short_status': 'מאושרת',
        'pl_area_dunam': 0.975,
    })
    assert 'שם התכנית: נוף הפארק' in meta
    assert 'סטטוס: מאושרת' in meta
    assert '0.975' in meta


def test_build_metadata_chunk_empty_attrs():
    assert MavatDownloader()._build_metadata_chunk({}) == ''


# ---------------------------------------------------------------- manager

def test_manager_instantiates_downloader_per_call(tmp_path, monkeypatch):
    instances = []

    class CountingDownloader:
        def __init__(self):
            instances.append(self)
            self.log = ['שורת לוג']
            self.metadata = []

        def download(self, plan_name, dest_dir):
            return []

    csv_path = tmp_path / 'sites.csv'
    csv_path.write_text('name,type,enabled\ntest,counting,true\n', encoding='utf-8')
    monkeypatch.setattr(manager_mod, 'SITES_CSV', csv_path)
    monkeypatch.setitem(manager_mod.DOWNLOADER_REGISTRY, 'counting', CountingDownloader)

    dm = manager_mod.DownloadManager(dest_dir=tmp_path)
    dm.download('תכנית א')
    dm.download('תכנית ב')
    # Regression: a single shared instance used to clobber concurrent logs.
    assert len(instances) == 2


def test_manager_returns_three_tuple(tmp_path, monkeypatch):
    class MetaDownloader:
        def __init__(self):
            self.log = ['לוג']
            self.metadata = ['[504] מטא']

        def download(self, plan_name, dest_dir):
            return []

    csv_path = tmp_path / 'sites.csv'
    csv_path.write_text('name,type,enabled\nm,meta,true\n', encoding='utf-8')
    monkeypatch.setattr(manager_mod, 'SITES_CSV', csv_path)
    monkeypatch.setitem(manager_mod.DOWNLOADER_REGISTRY, 'meta', MetaDownloader)

    files, log, metadata = manager_mod.DownloadManager(dest_dir=tmp_path).download('x')
    assert files == [] and log == ['לוג'] and metadata == ['[504] מטא']
