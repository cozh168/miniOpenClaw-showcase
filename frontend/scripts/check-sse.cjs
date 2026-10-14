const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const ts = require('typescript');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '..', 'src', 'lib', 'api.ts'), 'utf8');
const js = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020}}).outputText;
const context = {exports: {}, TextDecoder, Uint8Array, Error, fetch: async () => { throw new Error('unset fixture'); }};
vm.createContext(context); vm.runInContext(js, context);
async function runFixture(text, step = 10000) {
  const bytes = new TextEncoder().encode(text);
  context.fetch = async () => new Response(new ReadableStream({start(controller) {
    for (let i=0; i<bytes.length; i+=step) controller.enqueue(bytes.slice(i,i+step));
    controller.close();
  }}));
  const events = [];
  await context.exports.streamChat({message:'fixture', session_id:'fixture'}, {onEvent(event,data) {events.push([event,JSON.parse(JSON.stringify(data))]);}});
  return events;
}
(async () => {
  for (const ending of ['\n','\r\n','\r']) {
    const text = ['event: token','data: {"content":"中文"}','','event: done','data: {"content":"中文"}','',''].join(ending);
    for (const step of [1,7,10000]) assert.deepEqual(await runFixture(text,step), [['token',{content:'中文'}],['done',{content:'中文'}]]);
  }
  assert.deepEqual(await runFixture(': ping\r\nevent: token\r\ndata: {"content":\r\ndata: "多行"}\r\n\r\nevent: done\r\ndata: {}',1), [['token',{content:'多行'}],['done',{}]]);
  let cancelled = false;
  context.fetch = async () => new Response(new ReadableStream({start(c) {c.enqueue(new TextEncoder().encode('event: token\ndata: invalid\n\n'));},cancel() {cancelled = true;}}));
  await assert.rejects(context.exports.streamChat({message:'fixture',session_id:'fixture'}, {onEvent(){}}), /Unexpected token/);
  assert.equal(cancelled, true);
  console.log('PASS: 11 SSE regressions (LF/CRLF/CR × chunk sizes, UTF-8, multiline/final frame, malformed-frame cancellation).');
})().catch(error => {console.error(error); process.exitCode=1;});
