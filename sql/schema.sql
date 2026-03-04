-- ============================================================
-- HVAC Voice Agent — Supabase Schema
-- Run via Supabase SQL editor or psql.
-- Enable pgvector extension before running.
-- ============================================================

-- Required extensions
create extension if not exists "uuid-ossp";
create extension if not exists vector;

-- ─── calls ───────────────────────────────────────────────────────────────────
-- One row per inbound Twilio call.
create table if not exists calls (
    id                 text primary key,          -- "call_<twilio_call_sid_prefix>"
    twilio_call_sid    text unique not null,
    from_number        text not null,
    to_number          text,
    status             text not null default 'in_progress',
    -- status: in_progress | completed | emergency_escalated | abandoned
    outcome            text,
    started_at         timestamptz not null default now(),
    ended_at           timestamptz,
    duration_seconds   integer,
    is_emergency       boolean not null default false,
    priority           text not null default 'normal'  -- normal | urgent | emergency
);

-- ─── call_turns ──────────────────────────────────────────────────────────────
-- One row per message in the conversation.
create table if not exists call_turns (
    id         bigserial primary key,
    call_id    text not null references calls(id) on delete cascade,
    role       text not null,               -- user | assistant | system
    content    text not null,
    state      text,                        -- call state when this turn occurred
    created_at timestamptz not null default now()
);
create index if not exists idx_call_turns_call_id on call_turns(call_id);

-- ─── customers ───────────────────────────────────────────────────────────────
create table if not exists customers (
    id          uuid primary key default uuid_generate_v4(),
    phone       text unique not null,
    name        text,
    email       text,
    address     text,
    postal_code text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz
);
create index if not exists idx_customers_phone on customers(phone);

-- ─── bookings ────────────────────────────────────────────────────────────────
create table if not exists bookings (
    id                   text primary key,     -- "bk_YYYYMMDDHHmmss"
    call_id              text references calls(id),
    customer_name        text,
    phone                text,
    postal_code          text,
    issue_description    text,
    preferred_date       date,
    preferred_time_slot  text,                 -- e.g. "10:00-12:00"
    priority             text not null default 'normal',
    status               text not null default 'confirmed',
    -- status: confirmed | rescheduled | cancelled | completed
    cancellation_reason  text,
    technician_id        uuid,
    created_at           timestamptz not null default now(),
    updated_at           timestamptz
);
create index if not exists idx_bookings_phone on bookings(phone);
create index if not exists idx_bookings_date  on bookings(preferred_date);

-- Pricing tier: 'standard' (business hours) or 'surge' (after-hours)
alter table bookings add column if not exists pricing_tier text not null default 'standard';

-- Prevent double-booking: only one confirmed/rescheduled booking per slot
create unique index if not exists idx_bookings_slot_unique
    on bookings(preferred_date, preferred_time_slot)
    where status not in ('cancelled');

-- ─── escalations ─────────────────────────────────────────────────────────────
create table if not exists escalations (
    id                  bigserial primary key,
    call_id             text references calls(id),
    reason              text not null,
    transcript_summary  text,
    is_emergency        boolean not null default false,
    status              text not null default 'pending', -- pending | resolved
    resolved_by         text,
    resolved_at         timestamptz,
    created_at          timestamptz not null default now()
);
create index if not exists idx_escalations_status on escalations(status);

-- ─── rag_documents ───────────────────────────────────────────────────────────
-- Vector store for policy / knowledge base retrieval.
-- Only used when RAG is wired up; pgvector required.
create table if not exists rag_documents (
    id          bigserial primary key,
    title       text not null,
    content     text not null,
    source      text,                      -- file path or URL
    embedding   vector(1536),             -- OpenAI text-embedding-3-small dimensions
    created_at  timestamptz not null default now()
);
create index if not exists idx_rag_embedding
    on rag_documents using ivfflat (embedding vector_cosine_ops)
    with (lists = 50);

-- ─── Seed data — pre-booked slots for testing ────────────────────────────────
-- These represent existing bookings so the conflict-check and availability
-- fetch can be exercised.  Run once; ON CONFLICT DO NOTHING is safe to re-run.
insert into bookings (
    id, customer_name, phone, postal_code, issue_description,
    preferred_date, preferred_time_slot, priority, status, pricing_tier, created_at
) values
    ('bk_test001', 'Michael Chen',   '+14165550101', 'M5V 2K3', 'Furnace not heating',    '2026-03-05', '10:00-12:00', 'normal', 'confirmed', 'standard', now()),
    ('bk_test002', 'Sarah Williams', '+14165550102', 'M6K 1A1', 'AC unit noise',           '2026-03-05', '14:00-16:00', 'normal', 'confirmed', 'standard', now()),
    ('bk_test003', 'David Patel',    '+14165550103', 'M4B 1B3', 'Annual maintenance',      '2026-03-06', '09:00-11:00', 'normal', 'confirmed', 'standard', now()),
    ('bk_test004', 'Jennifer Kim',   '+14165550104', 'M2N 6A1', 'Thermostat replacement',  '2026-03-07', '10:00-12:00', 'normal', 'confirmed', 'standard', now()),
    ('bk_test005', 'Robert Singh',   '+14165550105', 'M3M 1N6', 'Emergency furnace repair','2026-03-05', '18:00-20:00', 'urgent', 'confirmed', 'surge',    now())
on conflict do nothing;
