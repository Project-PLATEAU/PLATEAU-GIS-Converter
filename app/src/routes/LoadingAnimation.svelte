<script lang="ts">
	import Icon from '@iconify/svelte';
	import { invoke } from '@tauri-apps/api/core';
	import { listen } from '@tauri-apps/api/event';
	import { onMount } from 'svelte';
	import VirtualScroll from 'svelte-virtual-scroll-list';

	async function cancelConversion() {
		await invoke('cancel_conversion');
	}

	type Item = {
		id: number;
		message: string;
		level: string;
		error_message?: string;
		source: string;
	};

	let items: Item[] = $state([]);

	let logView: VirtualScroll;

	// Setup log monitor
	onMount(() => {
		let promise = listen<{
			message: string;
			level: string;
			error_message?: string;
			source: string;
		}>('conversion-log', (event) => {
			items.push({
				id: items.length,
				...event.payload
			});
			items = items;
			logView.scrollToBottom();
		});

		return () => {
			promise.then((unlisten) => {
				unlisten();
			});
		};
	});
</script>

<div class="flex flex-col gap-4 p-12 place-items-center">
	<p class="text-white text-center font-semibold text-2xl">変換中 &hellip;</p>
	<div class="flex justify-center [&>div]:bg-white [&>div]">
		<div class="animate-loading-pulse-[-0.24s] bg-white h-4 w-4 rounded-full m-2"></div>
		<div class="animate-loading-pulse-[-0.12s] bg-white h-4 w-4 rounded-full m-2"></div>
		<div class="animate-loading-pulse-[0s] bg-white h-4 w-4 rounded-full m-2"></div>
	</div>

	<div
		class="my-5 w-full h-96 max-h-96 bg-slate-900/70 text-slate-300 text-xs font-mono p-1 rounded-sm"
	>
		<div class="w-full h-full">
			<VirtualScroll bind:this={logView} data={items} key="id">
				{#snippet children({ data }: { data: Item })}
					<div>
						{#if data.level === 'ERROR'}
							<span
								class="inline-flex items-center rounded-md bg-red-400/10 px-1 py-0.5 text-xs font-medium text-red-400 ring-1 ring-inset ring-red-400/20"
								>ERROR</span
							>
						{:else if data.level === 'WARN'}
							<span
								class="inline-flex items-center rounded-md bg-yellow-400/10 px-1 py-0.5 text-xs font-medium text-yellow-500 ring-1 ring-inset ring-yellow-400/20"
								>WARN</span
							>
						{:else if data.level === 'INFO'}
							<span
								class="inline-flex items-center rounded-md bg-blue-400/10 px-1 py-0.5 text-xs font-medium text-blue-400 ring-1 ring-inset ring-blue-400/30"
								>INFO</span
							>
						{/if}

						<span
							class="inline-flex items-center rounded-md bg-gray-400/10 px-1 py-0.5 text-xs font-medium text-gray-400 ring-1 ring-inset ring-gray-400/20"
							>{data.source}</span
						>

						{data.message}

						{#if data.error_message}
							<span class="text-red-600">{data.error_message}</span>
						{/if}
					</div>
				{/snippet}
			</VirtualScroll>
		</div>
	</div>

	<div>
		<button
			onclick={cancelConversion}
			class="bg-red-300 flex items-center font-bold py-1.5 pl-3 pr-5 rounded-full gap-1 shawdow-2xl hover:bg-red-400"
		>
			<Icon class="text-lg" icon="ic:baseline-cancel" />
			キャンセル
		</button>
	</div>
</div>
