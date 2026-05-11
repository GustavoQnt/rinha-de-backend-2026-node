import Fastify from 'fastify';
import { RESPONSE_BODIES } from './classifier.js';
import { createRuntimeClassifier } from './runtime-classifier.js';

const port = Number(process.env.PORT || 3000);
const host = process.env.HOST || '0.0.0.0';

const fastify = Fastify({
  logger: false,
  bodyLimit: 4096,
  disableRequestLogging: true,
  routerOptions: {
    ignoreTrailingSlash: true,
  },
});

const FALLBACK_BODY = RESPONSE_BODIES[0];
const classifierOptions = {};

if (process.env.CLASSIFIER === 'knn-bin') {
  const { vectorize } = await import('./vectorizer.js');
  classifierOptions.vectorize = vectorize;
}

const runtimeClassifier = createRuntimeClassifier(classifierOptions);

fastify.get('/ready', async (_request, reply) => {
  return reply.code(204).send();
});

fastify.post('/fraud-score', (request, reply) => {
  reply.header('content-type', 'application/json; charset=utf-8');
  try {
    reply.send(RESPONSE_BODIES[runtimeClassifier.classifyFraudBucket(request.body)]);
  } catch {
    reply.send(FALLBACK_BODY);
  }
});

await fastify.listen({ port, host });
