import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2";
import Anthropic from "npm:@anthropic-ai/sdk";

// ============================================================
// CORS
// ============================================================
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
  });
}

// ============================================================
// MODEL
// ============================================================
const MODEL = "claude-opus-5";

// ============================================================
// ENTRYPOINT
// ============================================================
Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: CORS_HEADERS });
  }
  if (req.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  // --- Auth: the gateway already rejected unauthenticated requests
  // (verify_jwt is enabled on this function), but we still need the raw
  // token here to build a Supabase client scoped to THIS user, so every
  // query below goes through PostgREST as them and is filtered by the
  // exact same row-level-security policies the rest of the app relies on.
  // We deliberately never use the service-role key in this function —
  // that would bypass RLS and could leak another organization's data.
  const authHeader = req.headers.get("Authorization");
  if (!authHeader) {
    return jsonResponse({ error: "Missing Authorization header" }, 401);
  }

  const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
  const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
  const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");

  if (!ANTHROPIC_API_KEY) {
    return jsonResponse({
      error: "AI assistant is not configured yet. Ask an administrator to set the ANTHROPIC_API_KEY secret for this project.",
    }, 503);
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: authHeader } },
    auth: { persistSession: false },
  });

  // Confirms the JWT is actually valid (not just present) and gives us
  // the user id for logging/prompts. Bad/expired tokens fail here.
  const { data: userData, error: userError } = await supabase.auth.getUser();
  if (userError || !userData?.user) {
    return jsonResponse({ error: "Invalid or expired session" }, 401);
  }
  const user = userData.user;

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON body" }, 400);
  }

  const action = typeof body.action === "string" ? body.action : "";
  const anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

  try {
    switch (action) {
      case "ask":
        return jsonResponse(await handleAsk(supabase, anthropic, body));
      case "summarize_lead":
        return jsonResponse(await handleSummarizeLead(supabase, anthropic, body));
      case "summarize_contact":
        return jsonResponse(await handleSummarizeContact(supabase, anthropic, body));
      case "summarize_deal":
        return jsonResponse(await handleSummarizeDeal(supabase, anthropic, body));
      case "score_lead":
        return jsonResponse(await handleScoreLead(supabase, anthropic, body));
      case "next_action":
        return jsonResponse(await handleNextAction(supabase, anthropic, body));
      case "draft_message":
        return jsonResponse(await handleDraftMessage(supabase, anthropic, body));
      case "meeting_summary":
        return jsonResponse(await handleMeetingSummary(supabase, anthropic, body));
      case "suggest_tasks":
        return jsonResponse(await handleSuggestTasks(supabase, anthropic, body));
      default:
        return jsonResponse({ error: "Unknown action: " + action }, 400);
    }
  } catch (err) {
    console.error("ai-assistant error", action, user.id, err);
    const message = err instanceof AppError ? err.message : "Er ging iets mis bij het genereren van het AI-antwoord.";
    const status = err instanceof AppError ? err.status : 500;
    return jsonResponse({ error: message }, status);
  }
});

// ============================================================
// Error helper (never leaks raw internals to the client)
// ============================================================
class AppError extends Error {
  status: number;
  constructor(message: string, status = 400) {
    super(message);
    this.status = status;
  }
}

// ============================================================
// Small helpers
// ============================================================
function textOf(response: Anthropic.Message): string {
  const block = response.content.find((b) => b.type === "text");
  return block && block.type === "text" ? block.text.trim() : "";
}

function parseJsonLoose<T>(raw: string): T {
  // Claude is instructed to return only JSON, but strip any accidental
  // markdown fencing defensively before parsing.
  const cleaned = raw.trim().replace(/^```(?:json)?/i, "").replace(/```$/i, "").trim();
  return JSON.parse(cleaned) as T;
}

const SYSTEM_PREFIX =
  "Je bent de AI-assistent binnen NextCRM, een CRM-systeem. Je antwoordt altijd in het Nederlands, " +
  "bent bondig en concreet, en baseert je uitsluitend op de gegevens die je hieronder krijgt. " +
  "Verzin nooit namen, bedragen of feiten die niet in de gegeven context staan. " +
  "Als de context leeg of onvoldoende is, zeg dat eerlijk in plaats van te gokken.";

async function claude(
  anthropic: Anthropic,
  system: string,
  userText: string,
  opts: { effort?: "low" | "medium" | "high"; maxTokens?: number } = {},
): Promise<string> {
  const response = await anthropic.messages.create({
    model: MODEL,
    max_tokens: opts.maxTokens ?? 1200,
    system: SYSTEM_PREFIX + "\n\n" + system,
    output_config: { effort: opts.effort ?? "medium" },
    messages: [{ role: "user", content: userText }],
  });
  return textOf(response);
}

function requireId(body: Record<string, unknown>, key: string): string {
  const v = body[key];
  if (typeof v !== "string" || !v) throw new AppError(`Missing "${key}"`, 400);
  return v;
}

// ============================================================
// Entity loaders (all RLS-scoped through `supabase`)
// ============================================================
async function loadLead(supabase: any, id: string) {
  const { data, error } = await supabase.from("leads").select("*").eq("id", id).maybeSingle();
  if (error) throw new AppError("Kon lead niet ophalen.", 500);
  if (!data) throw new AppError("Lead niet gevonden (of je hebt hier geen toegang toe).", 404);
  return data;
}
async function loadContact(supabase: any, id: string) {
  const { data, error } = await supabase.from("contacts").select("*").eq("id", id).maybeSingle();
  if (error) throw new AppError("Kon contact niet ophalen.", 500);
  if (!data) throw new AppError("Contact niet gevonden (of je hebt hier geen toegang toe).", 404);
  return data;
}
async function loadDeal(supabase: any, id: string) {
  const { data, error } = await supabase
    .from("deals")
    .select("*, stages(name), contacts(name, company, email)")
    .eq("id", id)
    .maybeSingle();
  if (error) throw new AppError("Kon deal niet ophalen.", 500);
  if (!data) throw new AppError("Deal niet gevonden (of je hebt hier geen toegang toe).", 404);
  return data;
}
async function loadNotesAndActivities(supabase: any, entityType: string, entityId: string) {
  const [{ data: notes }, { data: activities }] = await Promise.all([
    supabase.from("notes").select("body, created_at").eq("entity_type", entityType).eq("entity_id", entityId)
      .order("created_at", { ascending: false }).limit(10),
    supabase.from("activities").select("message, created_at").eq("entity_type", entityType).eq("entity_id", entityId)
      .order("created_at", { ascending: false }).limit(10),
  ]);
  return { notes: notes ?? [], activities: activities ?? [] };
}

// ============================================================
// 1/2/3: Lead / Contact / Deal summaries
// ============================================================
async function handleSummarizeLead(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const lead = await loadLead(supabase, requireId(body, "leadId"));
  const { notes, activities } = await loadNotesAndActivities(supabase, "lead", lead.id);
  const context = {
    naam: lead.name, bedrijf: lead.company, email: lead.email, telefoon: lead.phone,
    bron: lead.source, status: lead.status, score: lead.score, aangemaakt: lead.created_at,
    notities: notes.map((n: any) => n.body), activiteiten: activities.map((a: any) => a.message),
  };
  const summary = await claude(
    anthropic,
    "Vat de onderstaande lead samen in 3-5 zinnen voor een sales medewerker: wie het is, waar het vandaan komt, en hoe warm de lead lijkt op basis van de gegevens.",
    JSON.stringify(context, null, 2),
    { effort: "low", maxTokens: 500 },
  );
  return { summary };
}

async function handleSummarizeContact(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const contact = await loadContact(supabase, requireId(body, "contactId"));
  const [{ notes, activities }, { data: deals }] = await Promise.all([
    loadNotesAndActivities(supabase, "contact", contact.id),
    supabase.from("deals").select("company, amount, status, stage_id, stages(name)").eq("contact_id", contact.id),
  ]);
  const context = {
    naam: contact.name, bedrijf: contact.company, rol: contact.job_title, tag: contact.tag,
    email: contact.email, telefoon: contact.phone, klant_sinds: contact.created_at,
    deals: (deals ?? []).map((d: any) => ({ naam: d.company, bedrag: d.amount, status: d.status, fase: d.stages?.name })),
    notities: notes.map((n: any) => n.body), activiteiten: activities.map((a: any) => a.message),
  };
  const summary = await claude(
    anthropic,
    "Vat deze klant/contact samen in 3-5 zinnen: wie het is, de relatie tot nu toe (deals, historie), en waar iemand die dit contact voor het eerst opent op moet letten.",
    JSON.stringify(context, null, 2),
    { effort: "low", maxTokens: 500 },
  );
  return { summary };
}

async function handleSummarizeDeal(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const deal = await loadDeal(supabase, requireId(body, "dealId"));
  const { notes, activities } = await loadNotesAndActivities(supabase, "deal", deal.id);
  const context = {
    naam: deal.company, waarde: deal.amount, winkans: deal.probability, status: deal.status,
    fase: deal.stages?.name, verwachte_sluitdatum: deal.expected_close,
    contact: deal.contacts ? { naam: deal.contacts.name, bedrijf: deal.contacts.company } : null,
    notities: notes.map((n: any) => n.body), activiteiten: activities.map((a: any) => a.message),
  };
  const summary = await claude(
    anthropic,
    "Vat deze deal samen in 3-4 zinnen: status, kans van slagen, en het belangrijkste aandachtspunt om deze deal verder te helpen.",
    JSON.stringify(context, null, 2),
    { effort: "low", maxTokens: 400 },
  );
  return { summary };
}

// ============================================================
// 4: Lead scoring (suggestion only — never writes to the DB)
// ============================================================
async function handleScoreLead(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const lead = await loadLead(supabase, requireId(body, "leadId"));
  const { notes, activities } = await loadNotesAndActivities(supabase, "lead", lead.id);
  const context = {
    naam: lead.name, bedrijf: lead.company, bron: lead.source, status: lead.status,
    huidige_score: lead.score, notities: notes.map((n: any) => n.body), activiteiten: activities.map((a: any) => a.message),
  };
  const raw = await claude(
    anthropic,
    "Beoordeel hoe kansrijk deze lead is (0-100, waarbij 100 zeer kansrijk is) op basis van de gegevens. " +
      'Antwoord ALLEEN met geldige JSON in dit exacte formaat, zonder uitleg erbuiten: {"score": <getal 0-100>, "reasoning": "<een korte, concrete onderbouwing in 1-2 zinnen>"}',
    JSON.stringify(context, null, 2),
    { effort: "medium", maxTokens: 400 },
  );
  const parsed = parseJsonLoose<{ score: number; reasoning: string }>(raw);
  const score = Math.max(0, Math.min(100, Math.round(Number(parsed.score) || 0)));
  return { score, reasoning: String(parsed.reasoning || "") };
}

// ============================================================
// 5: Suggested next action (per entity)
// ============================================================
async function handleNextAction(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const entityType = String(body.entityType || "");
  const entityId = requireId(body, "entityId");
  let context: Record<string, unknown>;
  if (entityType === "lead") {
    const lead = await loadLead(supabase, entityId);
    const { notes, activities } = await loadNotesAndActivities(supabase, "lead", entityId);
    context = { type: "lead", naam: lead.name, bedrijf: lead.company, status: lead.status, score: lead.score, notities: notes, activiteiten: activities };
  } else if (entityType === "contact") {
    const contact = await loadContact(supabase, entityId);
    const { notes, activities } = await loadNotesAndActivities(supabase, "contact", entityId);
    context = { type: "contact", naam: contact.name, bedrijf: contact.company, tag: contact.tag, notities: notes, activiteiten: activities };
  } else if (entityType === "deal") {
    const deal = await loadDeal(supabase, entityId);
    const { notes, activities } = await loadNotesAndActivities(supabase, "deal", entityId);
    context = { type: "deal", naam: deal.company, waarde: deal.amount, fase: deal.stages?.name, status: deal.status, notities: notes, activiteiten: activities };
  } else {
    throw new AppError("Ongeldig entityType", 400);
  }
  const raw = await claude(
    anthropic,
    "Stel op basis van deze gegevens de EEN beste volgende actie voor. " +
      'Antwoord ALLEEN met geldige JSON: {"action": "<concrete actie in 1 zin>", "reasoning": "<korte onderbouwing>", "suggestedTask": {"title": "<taaknaam>", "priority": "hoog"|"normaal"|"laag", "dueInDays": <getal of null>}}',
    JSON.stringify(context, null, 2),
    { effort: "medium", maxTokens: 500 },
  );
  return parseJsonLoose<Record<string, unknown>>(raw);
}

// ============================================================
// 6/7: Draft message (follow-up note or full email)
// ============================================================
async function handleDraftMessage(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const entityType = String(body.entityType || "");
  const entityId = requireId(body, "entityId");
  const kind = body.kind === "email" ? "email" : "followup_note";
  const extraInstruction = typeof body.instruction === "string" ? body.instruction.slice(0, 500) : "";

  let contextObj: Record<string, unknown>;
  let recipientName = "";
  if (entityType === "contact") {
    const contact = await loadContact(supabase, entityId);
    recipientName = contact.name;
    contextObj = { naam: contact.name, bedrijf: contact.company, rol: contact.job_title, tag: contact.tag };
  } else if (entityType === "deal") {
    const deal = await loadDeal(supabase, entityId);
    recipientName = deal.contacts?.name || "";
    contextObj = { deal: deal.company, waarde: deal.amount, fase: deal.stages?.name, contact: deal.contacts?.name, bedrijf: deal.contacts?.company };
  } else if (entityType === "lead") {
    const lead = await loadLead(supabase, entityId);
    recipientName = lead.name;
    contextObj = { naam: lead.name, bedrijf: lead.company, status: lead.status };
  } else {
    throw new AppError("Ongeldig entityType", 400);
  }
  const { notes, activities } = await loadNotesAndActivities(supabase, entityType, entityId);
  (contextObj as any).notities = notes;
  (contextObj as any).activiteiten = activities;
  if (extraInstruction) (contextObj as any).extra_instructie_van_gebruiker = extraInstruction;

  if (kind === "email") {
    const raw = await claude(
      anthropic,
      "Schrijf een korte, professionele e-mail (in het Nederlands, geen overdreven formele toon) naar deze persoon, passend bij de context. " +
        'Antwoord ALLEEN met geldige JSON: {"subject": "<onderwerpregel>", "body": "<e-mailtekst met \\n voor nieuwe regels, zonder aanhef-placeholder zoals [naam] tenzij echt nodig>"}',
      JSON.stringify(contextObj, null, 2),
      { effort: "medium", maxTokens: 800 },
    );
    const parsed = parseJsonLoose<{ subject: string; body: string }>(raw);
    return { subject: String(parsed.subject || ""), body: String(parsed.body || ""), recipientName };
  }

  const message = await claude(
    anthropic,
    "Schrijf een korte follow-up notitie/bericht (2-4 zinnen, informeel-professioneel) die de gebruiker naar deze persoon kan sturen, passend bij de context.",
    JSON.stringify(contextObj, null, 2),
    { effort: "medium", maxTokens: 400 },
  );
  return { message, recipientName };
}

// ============================================================
// 8: Meeting summary (user-supplied transcript, not stored data)
// ============================================================
async function handleMeetingSummary(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const transcript = typeof body.transcript === "string" ? body.transcript.trim() : "";
  if (!transcript) throw new AppError("Vul eerst notities of een transcript in.", 400);
  if (transcript.length > 20000) throw new AppError("Tekst is te lang (max 20.000 tekens).", 400);

  const entityType = String(body.entityType || "");
  const entityId = typeof body.entityId === "string" ? body.entityId : "";
  let label = "";
  if (entityId && entityType === "contact") label = (await loadContact(supabase, entityId)).name;
  else if (entityId && entityType === "deal") label = (await loadDeal(supabase, entityId)).company;
  else if (entityId && entityType === "lead") label = (await loadLead(supabase, entityId)).name;

  const raw = await claude(
    anthropic,
    "Dit zijn ruwe notities of een transcript van een gesprek" + (label ? ` met/over "${label}"` : "") + ". " +
      "Maak er een beknopte samenvatting van (belangrijkste punten, besluiten) en stel 0-4 concrete vervolgtaken voor. " +
      'Antwoord ALLEEN met geldige JSON: {"summary": "<samenvatting, meerdere zinnen>", "actionItems": [{"title": "<taak>", "priority": "hoog"|"normaal"|"laag", "dueInDays": <getal of null>}]}',
    transcript,
    { effort: "medium", maxTokens: 1200 },
  );
  return parseJsonLoose<Record<string, unknown>>(raw);
}

// ============================================================
// 9: Task suggestions
// ============================================================
async function handleSuggestTasks(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const entityType = String(body.entityType || "");
  const entityId = requireId(body, "entityId");
  let contextObj: Record<string, unknown>;
  let existingTasksQuery;
  if (entityType === "contact") {
    const contact = await loadContact(supabase, entityId);
    contextObj = { type: "contact", naam: contact.name, bedrijf: contact.company, tag: contact.tag };
    existingTasksQuery = supabase.from("tasks").select("title, done").eq("contact_id", entityId);
  } else if (entityType === "deal") {
    const deal = await loadDeal(supabase, entityId);
    contextObj = { type: "deal", naam: deal.company, waarde: deal.amount, fase: deal.stages?.name, status: deal.status };
    existingTasksQuery = supabase.from("tasks").select("title, done").eq("deal_id", entityId);
  } else {
    throw new AppError("Ongeldig entityType", 400);
  }
  const [{ notes, activities }, { data: existingTasks }] = await Promise.all([
    loadNotesAndActivities(supabase, entityType, entityId),
    existingTasksQuery,
  ]);
  (contextObj as any).notities = notes;
  (contextObj as any).activiteiten = activities;
  (contextObj as any).bestaande_taken = existingTasks ?? [];

  const raw = await claude(
    anthropic,
    "Stel 2 tot 4 concrete, korte vervolgtaken voor die nu nog niet in de bestaande taken staan. " +
      'Antwoord ALLEEN met geldige JSON: {"tasks": [{"title": "<korte taaknaam>", "priority": "hoog"|"normaal"|"laag", "dueInDays": <getal of null>}]}',
    JSON.stringify(contextObj, null, 2),
    { effort: "medium", maxTokens: 600 },
  );
  return parseJsonLoose<{ tasks: unknown[] }>(raw);
}

// ============================================================
// 10 + example: natural-language ask / daily focus
// ============================================================
async function handleAsk(supabase: any, anthropic: Anthropic, body: Record<string, unknown>) {
  const query = typeof body.query === "string" ? body.query.trim() : "";
  if (!query) throw new AppError("Vul een vraag in.", 400);
  if (query.length > 2000) throw new AppError("Vraag is te lang.", 400);

  const today = new Date().toISOString().slice(0, 10);

  const [tasksRes, leadsRes, dealsRes, activitiesRes] = await Promise.all([
    supabase.from("tasks").select("title, priority, due_date, done, contacts(name), deals(company)")
      .eq("done", false).order("due_date", { ascending: true, nullsFirst: false }).limit(25),
    supabase.from("leads").select("name, company, status, score, created_at").neq("status", "converted")
      .order("score", { ascending: false }).limit(20),
    supabase.from("deals").select("company, amount, probability, status, expected_close, stages(name), contacts(name)")
      .eq("status", "open").order("expected_close", { ascending: true, nullsFirst: false }).limit(20),
    supabase.from("activities").select("message, created_at").order("created_at", { ascending: false }).limit(15),
  ]);

  const digest = {
    vandaag: today,
    openstaande_taken: (tasksRes.data ?? []).map((t: any) => ({
      titel: t.title, prioriteit: t.priority, deadline: t.due_date,
      gekoppeld_aan: t.deals?.company || t.contacts?.name || null,
    })),
    leads: (leadsRes.data ?? []).map((l: any) => ({ naam: l.name, bedrijf: l.company, status: l.status, score: l.score })),
    open_deals: (dealsRes.data ?? []).map((d: any) => ({
      naam: d.company, waarde: d.amount, winkans: d.probability, fase: d.stages?.name,
      verwachte_sluitdatum: d.expected_close, contact: d.contacts?.name,
    })),
    recente_activiteit: (activitiesRes.data ?? []).map((a: any) => a.message),
  };

  const answer = await claude(
    anthropic,
    "Je krijgt hieronder een momentopname van de openstaande taken, leads, open deals en recente activiteit van de gebruiker in het CRM. " +
      "Beantwoord de vraag van de gebruiker met concrete, uitvoerbare aanbevelingen die verwijzen naar specifieke namen/bedrijven uit de gegevens. " +
      "Als de gebruiker vraagt waar ze zich vandaag op moeten focussen, prioriteer op: verlopen/dichtbije deadlines, hoogst scorende leads, en deals die al lang stilliggen. " +
      "Gebruik korte alinea's of een bullet-lijst, geen lange lappen tekst.",
    "CRM-gegevens:\n" + JSON.stringify(digest, null, 2) + "\n\nVraag van de gebruiker: " + query,
    { effort: "high", maxTokens: 1200 },
  );
  return { answer };
}
