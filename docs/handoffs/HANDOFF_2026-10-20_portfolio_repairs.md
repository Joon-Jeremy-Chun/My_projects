# 인계 — My_projects 포트폴리오·미완성·오류 보완

날짜: 2026-10-20 · 담당: Codex.

## 요청과 승인

[팩트] 사용자는 이전 저장소처럼 상세 정리를 요청한 뒤 미완성·오류 수정과 게시를
명시적으로 요청했다. 문서·코드 보완과 검증한 결과의 commit/push가 승인된 범위다.
계좌 작업, 실제 드론 비행, 새 API 수집이나 전체 학습을 요청한 것은 아니다.

## 작업 위치와 Git 경계

- 원래 My_projects의 로컬 master 기준은 ed71db0다.
- 당시 원격 master는 58a5121이고 로컬에 14커밋이 더 있었다.
- 원래 작업 폴더의 수정·삭제 상태 파일과 미추적 파일 149개는 보존한다.
- 수정은 별도 my_projects-maintenance-20261020 worktree의
  codex/portfolio-repair-20261020 브랜치에서 준비했다.
- 기존 코드·그림·데이터를 이동/삭제하거나 출력 전체를 재생성하지 않았다.

## 결과

- 루트 1개, 분야 16개, 상세 연구/연습 43개로 README 총 60개.
- 성과 근거표, 수정 기록, 재현 안내, 주제 코드 306개 전체 목록.
- 직접 작성한 SVG 표지 17개와 기존 그림을 연결한 출처 문서.
- 미완성 Python 4개를 SIS·확산·사회비용·Week 5 시각화 참조 구현으로 완성.
- 기존 Python 19개 수정: 문법/이름, 데이터 경로, 흐름식, 나이 집계, CGS,
  BVP 부호, Simpson, Crank–Nicolson 열원, hybrid bracket, GUE KS 비교 등.
- 전용 회귀 테스트 21개 통과; Python 전체 critical 정적 검사 통과.
- 공용·Claude·Codex 지침, 공유 기록, 도구 및 검증 의존성 안내 추가.

개별 파일과 정확한 검증 범위는 [REPAIRS.md](../REPAIRS.md)를 따른다.

## 게시 시 중요한 조건

[팩트] 과거 미게시 이력에서 최대 636,205,778바이트 CSV를 포함한 대용량 객체를
확인했다. 현재 최종 트리에는 해당 파일이 없고, 현재 기존 추적 파일의 최대 크기는
13,541,410바이트다. GitHub는 100MiB를 넘는 일반 Git 파일을 차단한다.
[공식 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

[추론] 기존 로컬 14커밋을 보존하고 원격 과거 이력을 유지하면서 게시하려면,
origin/master 위의 별도 게시 브랜치에 검토한 최종 트리의 diff를 적용해 새 커밋으로
전달하는 것이 적절하다. 이 방법은 기존 원격 커밋을 재작성하거나 force-push하지 않는다.
큰 과거 객체를 원격으로 전송하지 않고 현재 완성된 파일 상태를 전달한다.

원래 로컬 master를 그대로 push하면 대용량 과거 객체가 다시 포함될 수 있다.
후속 작업은 게시 완료 후 확인한 원격/게시 브랜치를 기준으로 시작한다.
원래 사용자 작업에 reset, rebase, clean, 강제 checkout을 적용하지 않는다.

## 검증과 해석의 한계

- Python AST, 노트북 JSON, 상대 링크·그림·앵커, SVG 구조와 diff를 검사했다.
- 최종 로컬 구조 검사: Python 291개, 노트북 JSON 5개, Markdown 74개,
  상대 참조 1,539개, SVG 17개가 통과했다. 주제 코드 306개가 목록에 모두 포함됐다.
- 원래 작업 폴더의 기준 파일 hash를 대조해 기존 파일 변경이 없음을 확인했다.
- 루트 데스크톱/좁은 화면, 선형대수·제타·학습 자료·논문 안내와 성과 표를 임시
  HTML로 렌더링해 확인했다. Mermaid는 소스 확인이며 실제 원격 렌더링은 별도다.
- 테스트 출력 그림은 임시 디렉터리에만 썼다. 기존 PNG/GIF/CSV/노트북 출력 변경은 없다.
- 과거 SPORA 비용·time-grid 가정은 그대로 설명했다. 새 흐름 기반 비용 helper를
  논문 수치의 재검증이나 교체 결과로 주장하지 않는다.
- 모든 노트북 실행·전체 최적화·학습·R 실행·API·broker·드론 작업은 수행하지 않았다.

## 완료 확인

[팩트] 문서·코드 수정과 검증을 완료하고 GitHub master에 게시했다.

- 수정 보관 커밋: `d1dbfc7f7dafad54cd193df592ec71d9d9a2d4df`.
- 구현·문서 게시 커밋: `5b2833eb68a75399a9239b816ec6206b66f5e9be`.
- 두 커밋의 동일한 트리: `4410cd1774f8fb26a245a34962a6776e521e0c42`.
- 게시 worktree: `my_projects-publish-20261020`.
- 게시 브랜치: `codex/portfolio-publish-20261020`.
- 일반 push 성공 후 `git ls-remote --heads origin master`와 로컬 SHA 일치 확인.
- 기존 원격 `58a5121`의 fast-forward이며 원래 로컬 14커밋과 미커밋 작업 보존.

현재 완료 기록은 위 게시 이후의 별도 문서 커밋으로 남긴다. 후속 작업은 원격 master
또는 게시 브랜치에서 시작한다. 원래 master를 원격에 맞추려고 초기화하지 않는다.

### 최종 트리 전달 시 참고

전역 PDF textconv가 활성화되어 있어 단순 `git diff --binary`는 PDF를 텍스트로 바꾼다.
적용 가능한 원본 binary diff는 `--no-ext-diff --no-textconv --binary --full-index`로 만들었다.
패치 적용 후 검토한 수정 커밋과 트리 해시가 완전히 같은지 확인한 뒤 게시했다.
검토한 수정의 공백 검사는 통과했다. 이전 미게시 커밋에 이미 있던 공백은 보존했으므로
이전 원격 전체와의 차이에 대한 `diff --check`는 기존 공백을 보고할 수 있다.
